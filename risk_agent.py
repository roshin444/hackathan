import pandas as pd
import json
import ast
import re
from langchain.schema import SystemMessage, HumanMessage
from config import llm

class RiskAssessmentAgent:
    def __init__(self, findings_path_or_df, checklist_path_or_df, criteria_path_or_dict):
        # --- Audit findings ---
        if isinstance(findings_path_or_df, pd.DataFrame):
            self.findings = findings_path_or_df
        else:
            self.findings = pd.read_csv(findings_path_or_df)

        # --- Compliance checklist ---
        if isinstance(checklist_path_or_df, pd.DataFrame):
            self.checklist = checklist_path_or_df
        else:
            self.checklist = pd.read_csv(checklist_path_or_df)

        # --- Risk criteria ---
        if isinstance(criteria_path_or_dict, dict):
            self.criteria = criteria_path_or_dict
        else:
            self.criteria = json.load(criteria_path_or_dict)

        # Merge and standardize
        self.df = self._merge_and_standardize()

    def _standardize_columns(self, df, file_type="audit"):
        """Map arbitrary columns to a generic schema"""
        mapping = {}

        for col in df.columns:
            name = col.lower()
            if any(x in name for x in ["id", "finding", "issue"]):
                mapping[col] = "finding_id"
            elif any(x in name for x in ["description", "finding", "text"]):
                mapping[col] = "description"
            elif any(x in name for x in ["severity", "risk", "impact"]):
                mapping[col] = "severity"
            elif any(x in name for x in ["status", "state"]):
                mapping[col] = "status"
            elif any(x in name for x in ["control", "control_id"]):
                mapping[col] = "control_id"
            elif any(x in name for x in ["framework", "standard"]):
                mapping[col] = "framework"
            else:
                mapping[col] = "additional_info"

        df = df.rename(columns=mapping)
        return df

    def _merge_and_standardize(self):
        """Merge audit + compliance CSVs into a unified DataFrame safely"""
        audit_df = self._standardize_columns(self.findings, "audit")
        checklist_df = self._standardize_columns(self.checklist, "checklist")

        # Remove any duplicate columns within each DataFrame
        audit_df = audit_df.loc[:, ~audit_df.columns.duplicated()]
        checklist_df = checklist_df.loc[:, ~checklist_df.columns.duplicated()]

        if "control_id" in audit_df.columns and "control_id" in checklist_df.columns:
            # Merge on control_id with suffixes
            merged = pd.merge(
                audit_df, checklist_df, on="control_id", how="outer", suffixes=("_audit", "_checklist")
            )
        else:
            # Add suffixes to avoid duplicate column names
            audit_df = audit_df.add_suffix("_audit")
            checklist_df = checklist_df.add_suffix("_checklist")
            merged = pd.concat([audit_df, checklist_df], ignore_index=True, sort=False)

        return merged

    def generate_summary(self):
        """Generate structured risk assessment JSON for any CSV combination"""
        system_msg = SystemMessage(content="""
You are an audit & compliance risk assessment expert.
Analyze the unified report DataFrame.
Return a JSON array with:
- All original columns from merged CSVs
- risk_level (Low/Medium/High/Critical)
- likelihood (1-5)
- impact (1-5)
- recommendation
Do not return any text outside JSON.
""")

        # Convert merged DataFrame to text for LLM context
        df_text = "\n".join([
            ", ".join([f"{col}: {self.df.iloc[i][col]}" for col in self.df.columns])
            for i in range(len(self.df))
        ])

        human_msg = HumanMessage(content=f"""
Unified Findings:
{df_text}

Risk Criteria:
{json.dumps(self.criteria) if self.criteria else "No criteria provided"}

Task: Assess risk for each finding, prioritize High → Low, and provide mitigation recommendations.
Return output as a valid JSON array.
""")

        messages = [system_msg, human_msg]
        response = llm.invoke(messages)

        # Robust JSON extraction
        response_text = response.content
        match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                output = json.loads(json_str)
            except json.JSONDecodeError:
                output = ast.literal_eval(json_str)
        else:
            output = [{"error": "Could not parse LLM response", "raw": response_text}]

        return output
