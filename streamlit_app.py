import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import glob
import os
from typing import Optional, List, Dict, Any, Tuple

# Import for the custom color scale styling
from matplotlib.colors import LinearSegmentedColormap, to_hex

# --- 1. CONFIGURATION & FILE PATHS (SET THIS UP FIRST!) ---
# ⚠️ IMPORTANT: Update this path to your actual FTP folder location ⚠️
BASE_PATH = 'C:/Users/HFreeman/Desktop/FTP/'

POLICY_FILE_PATTERN = os.path.join(BASE_PATH, 'Policies_DBGA_*.csv')
NSF_FILE_PATTERN = os.path.join(BASE_PATH, 'NSF_Activity_DBGA_*.csv')
CANCELLED_FILE_PATTERN = os.path.join(BASE_PATH, 'Cancelled_*.csv')

ANALYSIS_END_DATE = pd.to_datetime('2025-11-07')

# Suppress the SettingWithCopyWarning
warnings.filterwarnings('ignore', message='A value is trying to be set on a copy of a slice from a DataFrame.')

# Underwriting Class Mapping
uw_class_mapping: Dict[str, str] = {
    'W12GI': 'GI', 'W12GI FL': 'GI', 'W12GI KS': 'GI', 'W12GI KY': 'GI', 'W12GI MO': 'GI',
    'W12GI MT': 'GI', 'W12GI PA': 'GI', 'W12GI SC': 'GI', 'W12GI TX': 'GI', 'W12GI WI': 'GI',
    'W12GI WY': 'GI', 'WL12G': 'Graded', 'WL12G FL': 'Graded', 'WL12G KS': 'Graded',
    'WL12G KY': 'Graded', 'WL12G MO': 'Graded', 'WL12G MT': 'Graded', 'WL12G PA': 'Graded',
    'WL12G SC': 'Graded', 'WL12G TX': 'Graded', 'WL12G WI': 'Graded', 'WL12G WY': 'Graded',
    'WL12P': 'Preferred', 'WL12P FL': 'Preferred', 'WL12P KS': 'Preferred', 'WL12P KY': 'Preferred',
    'WL12P MO': 'Preferred', 'WL12P MT': 'Preferred', 'WL12P PA': 'Preferred', 'WL12P SC': 'Preferred',
    'WL12P TX': 'Preferred', 'WL12P WI': 'Preferred', 'WL12P WY': 'Preferred', 'WL12S': 'Standard',
    'WL12S FL': 'Standard', 'WL12S KS': 'Standard', 'WL12S KY': 'Standard', 'WL12S MO': 'Standard',
    'WL12S MT': 'Standard', 'WL12S PA': 'Standard', 'WL12S SC': 'Standard', 'WL12S TX': 'Standard',
    'WL12S WI': 'Standard', 'WL12S WY': 'Standard'
}
MAPPING_PLAN_CODES = list(uw_class_mapping.keys())

# Reason Definitions
# REASON_MAP is kept as it's used in assign_termination_reason which helps determine the 'Lapsed' status detail
REASON_MAP = {
    r'(?i)Check Returned': 'Check Returned',
    r'(?i)NSF \(OTHER|NSF': 'NSF',
    r'(?i)Account Closed': 'Account Closed',
    r'(?i)Credit Card Declined': 'Credit Card Declined',
    r'(?i)Stop Payment': 'Stop Payment'
}

# Type hint for the complex return tuple (reasons_df is removed)
AnalysisResult = Tuple[pd.DataFrame, int, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]

# --- 2. LOCAL FILE HANDLERS AND MERGE ---

def get_most_recent_file(pattern: str) -> Optional[str]:
    """ Finds the most recently modified file matching a pattern. """
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def clean_policy_nbr(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """ Standardizes policy_nbr column to 'policy_nbr' and cleans data. """
    if col_name in df.columns:
        df['policy_nbr'] = df[col_name].astype(str).str.strip()
        if col_name != 'policy_nbr':
            df = df.drop(columns=[col_name], errors='ignore')
    return df

def map_nsf_reason(df: pd.DataFrame) -> pd.DataFrame:
    """ Maps detailed NSF reasons to the simplified list. """
    df['Final_Reason'] = 'NSF: Unknown'
    for pattern, reason in REASON_MAP.items():
        mask = df['Clean_Reason'].astype(str).str.contains(pattern, case=False, na=False)
        df.loc[mask, 'Final_Reason'] = reason
    return df

@st.cache_data
def load_and_merge_data(
    policy_pattern: str, 
    nsf_pattern: str, 
    cancelled_pattern: str,
    cache_buster: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.Timestamp]: # Modified return type to include earliest_date
    """ 
    Loads all files from the local directory, cleans policy numbers, merges, and determines the final reason.
    FIXES:
    - Explicitly converts date columns to string before parsing.
    - Explicitly parses 'issue_date' and 'term_date' (now 'term_date', 'issue_date') as YYYYMMDD.
    """
    
    # Implementation change: always load the bundled `randompolicydata.csv` in the repo directory
    csv_path = os.path.join(os.path.dirname(__file__), 'randompolicydata.csv')

    if not os.path.exists(csv_path):
        # Return empty with NaT so callers can show the existing error message
        return pd.DataFrame(), pd.NaT

    # Read the bundled CSV
    df_policies = pd.read_csv(csv_path, encoding='unicode_escape', sep=',', on_bad_lines='skip', engine='python')
    
    # Normalize column names - This handles the 'Issue Date' -> 'issue_date' and 'Term Date' -> 'term_date' transformation.
    df_policies.columns = df_policies.columns.str.strip().str.replace(r'[^\w\s]', '', regex=True).str.lower().str.replace(r'\s+', '_', regex=True)
    
    df_policies = clean_policy_nbr(df_policies, col_name='policy_nbr')

    context_cols_to_keep = ['policy_nbr', 'term_date', 'issue_date', 'plan_code', 'tobacco', 'gender', 'app_recvd_date']
    df_policies = df_policies[[col for col in context_cols_to_keep if col in df_policies.columns]]

    # --- FIX 1 & 4: Robust Date Parsing ---
    date_cols_to_parse = ['app_recvd_date', 'issue_date', 'term_date']
    earliest_date = pd.NaT
    
    for col in date_cols_to_parse:
        if col in df_policies.columns:
            # FIX: Ensure dates are strings before parsing (handles numeric YYYYMMDD being read as int/float)
            df_policies[col] = df_policies[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            
            # FIX: Use the correct format for YYYYMMDD
            df_policies[col] = pd.to_datetime(df_policies[col], format='%Y%m%d', errors='coerce')
            
            # Debugging check:
            # st.write(f"Parsed {col}: {df_policies[col].notna().sum()} valid dates")


    # Compute earliest date
    if 'app_recvd_date' in df_policies.columns:
        earliest_date = df_policies['app_recvd_date'].min()
    
    # --- End Date Parsing Fixes ---

    if 'plan_code' in df_policies.columns:
        df_policies['plan_code'] = df_policies['plan_code'].astype(str).str.strip().str.upper()

    # There is no NSF/Cancelled merging when using the bundled CSV; set a default clean reason
    df_policies['clean_reason'] = 'Unknown'

    # DEBUGGING: Print data quality stats
    # st.write(f"Total records after load: {len(df_policies):,}")
    # st.write(f"Valid App Recvd Date: {df_policies['app_recvd_date'].notna().sum():,}")
    # st.write(f"Valid Issue Date: {df_policies['issue_date'].notna().sum():,}")
    # st.write(f"Valid Term Date: {df_policies['term_date'].notna().sum():,}")

    return df_policies, earliest_date


def assign_termination_reason(df: pd.DataFrame) -> pd.DataFrame:
    """ Assigns the final Termination_Reason based on the merged 'clean_reason' and Lapsed status. """
    # Keep this function as 'Termination_Reason' is used to identify terminated policies
    # even if the detailed analysis is removed.
    df['Termination_Reason'] = df['clean_reason'].where(
        df['Lapsed'] == 1, 
        'Active/Not Lapsed'
    )
    for col in ['Termination_Reason', 'clean_reason']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    for pattern, reason in REASON_MAP.items():
        mask = df['Termination_Reason'].astype(str).str.contains(pattern, case=False, na=False)
        df.loc[mask, 'Termination_Reason'] = reason
            
    return df

def calculate_initial_mixes(df_cohort: pd.DataFrame) -> pd.DataFrame:
    """ Calculates the initial policy mix (proportions) for each cohort. """
    mix_data = []
    
    if df_cohort.empty:
        return pd.DataFrame()
    
    df_cohort['Initial_Mix_Group'] = 1 # Temporary column for grouping
    
    for segment_col in ['underwriting_class', 'gender', 'tobacco']:
        if segment_col in df_cohort.columns:
            
            # Count policies by Cohort and Segment
            counts = df_cohort.groupby(['Cohort_Week_Index', segment_col])['policy_nbr'].nunique().reset_index(name='Policy_Count')
            
            # Calculate total policies in each Cohort
            totals = df_cohort.groupby('Cohort_Week_Index')['policy_nbr'].nunique().reset_index(name='Total_Cohort_Policies')
            
            # Merge to calculate proportion
            counts = pd.merge(counts, totals, on='Cohort_Week_Index')
            counts['Segment'] = segment_col.replace('_', ' ').title()
            counts['Proportion'] = (counts['Policy_Count'] / counts['Total_Cohort_Policies']) * 100
            
            counts.rename(columns={segment_col: 'Category'}, inplace=True)
            mix_data.append(counts)
            
    if mix_data:
        return pd.concat(mix_data, ignore_index=True)
    return pd.DataFrame()

def calculate_average_months_held(df: pd.DataFrame, analysis_end_date: pd.Timestamp, cohort_dates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the actual months a policy was held until termination or the analysis end date.
    Returns a DataFrame with segmented averages for each cohort.
    """
    df_calc = df.copy()

    # Calculate actual held duration in days
    df_calc['Held_End_Date'] = df_calc['term_date'].where(df_calc['Lapsed'] == 1, analysis_end_date)
    # FIX: Ensure 'issue_date' is a datetime type before subtraction
    if 'issue_date' in df_calc.columns and pd.api.types.is_datetime64_any_dtype(df_calc['issue_date']):
        df_calc['Held_Duration_Days'] = (df_calc['Held_End_Date'] - df_calc['issue_date']).dt.days
    else:
        # Fallback if issue_date is still problematic
        df_calc['Held_Duration_Days'] = 0 
        
    # Convert days to months
    df_calc['Months_Held'] = (df_calc['Held_Duration_Days'] / 30.4375).clip(lower=0)

    # Define all grouping columns and filter for existence
    grouping_cols = ['Cohort_Week_Index', 'underwriting_class', 'gender', 'tobacco']
    available_cols = [col for col in grouping_cols if col in df_calc.columns]

    segment_data = []
    
    # Calculate Averages by Cohort and Segment
    for cohort_index in df_calc['Cohort_Week_Index'].unique():
        df_cohort = df_calc[df_calc['Cohort_Week_Index'] == cohort_index]
        # Use .get() with a default in case the index is somehow missing (safer)
        cohort_start_date = cohort_dates_df.loc[cohort_index, 'Cohort_Start_Date'] if cohort_index in cohort_dates_df.index else pd.NaT
        
        # 1. Cohort Overall Average
        overall_avg = df_cohort['Months_Held'].mean()
        segment_data.append({
            'Cohort_Index': cohort_index,
            'Cohort_Date': cohort_start_date,
            'Segment': 'Cohort Overall', 
            'Category': 'Total', 
            'Avg_Months_Held': overall_avg, 
            'Policies': len(df_cohort)
        })

        # 2. Segment Averages within the Cohort
        for col in available_cols:
            if col != 'Cohort_Week_Index':
                segment_avg = df_cohort.groupby(col)['Months_Held'].agg(['mean', 'count']).reset_index()
                segment_avg.columns = ['Category', 'Avg_Months_Held', 'Policies']
                segment_avg['Segment'] = col.replace('_', ' ').title()
                segment_avg['Cohort_Index'] = cohort_index
                segment_avg['Cohort_Date'] = cohort_start_date
                segment_data.extend(segment_avg.to_dict('records'))

    return pd.DataFrame(segment_data)


@st.cache_data
def perform_cohort_analysis(
    df: pd.DataFrame, 
    cohort_start_date: pd.Timestamp,
    # The filter now accepts a generic list of selected plan codes
    selected_plans: list[str],
    selected_tobacco: list[str],
    selected_gender: list[str],
    selected_case_status: list[str] 
) -> AnalysisResult: # MODIFIED: Removed pd.DataFrame for reasons_df
    """ 
    Performs the full weekly cohort lapse analysis, now including Average Months Held and Mix calculation.
    FIX: Removed redundant date parsing for issue_date/term_date since it's now handled robustly in load_and_merge_data.
    """
    
    df_filtered = df.copy()

    # --- Data Cleaning and Filtering ---
    if 'plan_code' not in df_filtered.columns or 'policy_nbr' not in df_filtered.columns:
        # Return the extended tuple with empty dataframes
        return pd.DataFrame(), 0, pd.Series(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # MODIFIED: Removed empty DF for reasons_df

    # Apply the plan code filter immediately
    if selected_plans:
        df_filtered = df_filtered[df_filtered['plan_code'].isin(selected_plans)].copy()
        
    # Map to underwriting class *after* filtering (unmapped codes will result in NaN, which is fine)
    # Mapping happens here because we need it for segmentation charts later
    df_filtered['underwriting_class'] = df_filtered['plan_code'].map(uw_class_mapping)

    # Date check: App_recvd_date should be a datetime from load_and_merge_data
    if not pd.api.types.is_datetime64_any_dtype(df_filtered['app_recvd_date']):
        # Fallback to re-parse if needed, though load_and_merge_data should handle it.
        df_filtered['app_recvd_date'] = pd.to_datetime(df_filtered['app_recvd_date'], errors='coerce')


    # --- Existing Filters (Plan code filter moved up) ---
    if selected_tobacco and 'tobacco' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['tobacco'].isin(selected_tobacco)]
    if selected_gender and 'gender' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['gender'].isin(selected_gender)]

    df_cohort = df_filtered[df_filtered['app_recvd_date'] >= cohort_start_date].copy()
    
    if df_cohort.empty:
        # Return the extended tuple with empty dataframes
        return pd.DataFrame(), 0, pd.Series(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # MODIFIED: Removed empty DF for reasons_df

    df_cohort['Cohort_Week_Index'] = ((df_cohort['app_recvd_date'] - cohort_start_date).dt.days // 7 + 1)
    cohort_dates = df_cohort.groupby('Cohort_Week_Index')['app_recvd_date'].min().rename('Cohort_Start_Date')

    # --- Lapse Status and Tenure Calculation ---
    # FIX 3: Robust Lapsed Policy Check
    # A policy is Lapsed (1) if term_date is NOT NaT AND term_date is before or equal to the ANALYSIS_END_DATE
    df_cohort['Lapsed'] = (df_cohort['term_date'].notna() & (df_cohort['term_date'] <= ANALYSIS_END_DATE)).astype(int)
    
    # DEBUGGING: Check lapsed policy count
    # st.write(f"Total policies in analysis: {len(df_cohort):,}")
    # st.write(f"Lapsed policies: {df_cohort['Lapsed'].sum():,}")
    
    # --- Apply the Case Status Filter ---
    df_cohort['Case_Status'] = df_cohort['Lapsed'].apply(lambda x: 'Inactive' if x == 1 else 'Active')
    
    if selected_case_status:
        df_cohort = df_cohort[df_cohort['Case_Status'].isin(selected_case_status)].copy()
        
    if df_cohort.empty:
        # After filtering by status, we might have an empty DataFrame again
        return pd.DataFrame(), 0, pd.Series(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # MODIFIED: Removed empty DF for reasons_df

    # Recalculate Lapsed status based on the *filtered* set (important for charts)
    df_cohort['Lapsed'] = (df_cohort['term_date'].notna() & (df_cohort['term_date'] <= ANALYSIS_END_DATE)).astype(int)
    
    df_cohort['Actual_Lapse_Date'] = df_cohort['term_date'].where(df_cohort['Lapsed'] == 1, pd.NaT)
    
    # Calculate Policy Month for cross-cohort analysis (REASON ANALYSIS IS REMOVED, BUT KEEPING Policy_Month FOR REFERENCE IF NEEDED)
    # FIX: Ensure issue_date is valid before calculation
    if 'issue_date' in df_cohort.columns and pd.api.types.is_datetime64_any_dtype(df_cohort['issue_date']):
        df_cohort['Policy_Month'] = (((df_cohort['Actual_Lapse_Date'] - df_cohort['issue_date']).dt.days / 30.4375).fillna(0).astype(int) + 1).clip(upper=100)
    else:
        df_cohort['Policy_Month'] = 0
        
    df_cohort['Effective_End_Date'] = df_cohort['Actual_Lapse_Date'].fillna(ANALYSIS_END_DATE)
    df_cohort['Tenure_Weeks'] = ((df_cohort['Effective_End_Date'] - df_cohort['app_recvd_date']).dt.days / 7)
    df_cohort['Lapse_Week'] = ((df_cohort['Actual_Lapse_Date'] - df_cohort['app_recvd_date']).dt.days // 7 + 1)
    max_overall_tenure_week = int(df_cohort['Tenure_Weeks'].max()) + 1

    # --- ASSIGN TERMINATION REASONS ---
    df_cohort = assign_termination_reason(df_cohort) # Keep this for the Policy Detail Table if the user wants it.

    # --- Cohort Lapse Matrix Calculation (for Retention Plot) ---
    cohort_sizes = df_cohort.groupby('Cohort_Week_Index')['policy_nbr'].nunique().rename('Total_Policies')
    total_lapsed_policies = df_cohort[df_cohort['Lapsed'] == 1]['policy_nbr'].nunique()

    # DEBUGGING: Check total lapsed policies
    # st.write(f"Total lapsed policies after filters: {total_lapsed_policies:,}")

    lapse_counts = (df_cohort[df_cohort['Lapsed'] == 1].groupby(['Cohort_Week_Index', 'Lapse_Week'])['policy_nbr'].nunique().rename('Lapses'))
    lapse_matrix = lapse_counts.reset_index().pivot_table(index='Cohort_Week_Index', columns='Lapse_Week', values='Lapses', fill_value=0)
    
    all_lapse_weeks = pd.Index(range(1, max_overall_tenure_week + 1), name='Lapse_Week')
    lapse_matrix = lapse_matrix.reindex(columns=all_lapse_weeks, fill_value=0)
    cumulative_lapses = lapse_matrix.cumsum(axis=1)
    
    # Handle division by zero if a cohort has zero policies after filtering
    cumulative_lapse_rate = cumulative_lapses.div(cohort_sizes, axis=0) * 100
    retention_rate = 100 - cumulative_lapse_rate 
    
    # --- Dynamic Masking ---
    retention_rate.columns = [f'Week_{int(w)}' for w in retention_rate.columns]
    max_cohort_tenure = df_cohort.groupby('Cohort_Week_Index')['Tenure_Weeks'].max()
    mask = pd.DataFrame(False, index=retention_rate.index, columns=retention_rate.columns)
    week_indices = np.array([int(col.split('_')[1]) for col in retention_rate.columns])
    for cohort_index in retention_rate.index:
        tenure_threshold = max_cohort_tenure.loc[cohort_index]
        mask.loc[cohort_index] = week_indices > np.floor(tenure_threshold)
    retention_rate[mask] = np.nan
    retention_rate.index.name = 'Cohort_Week_Index'
    
    # --- Data Preparation for Plots/Tables & Persistency ---
    mix_df = df_cohort[['Cohort_Week_Index', 'Tenure_Weeks', 'Lapsed', 'Lapse_Week', 'underwriting_class', 'gender', 'tobacco', 'issue_date', 'term_date']].copy()
    
    # CALCULATE AVERAGE MONTHS HELD
    cohort_dates_map = cohort_dates.to_frame()
    avg_months_held_df = calculate_average_months_held(df_cohort, ANALYSIS_END_DATE, cohort_dates_map)
    
    # CALCULATE INITIAL MIXES
    initial_mix_df = calculate_initial_mixes(df_cohort)
    
    # reasons_df REMOVAL: Only keep basic details for terminated policies for the 'Detailed List' table if needed,
    # but rename it to be less confusing and remove reason-specific logic.
    policy_detail_cols = ['policy_nbr', 'term_date', 'issue_date', 'plan_code', 'tobacco', 'gender']
    terminated_detail_df = df_cohort[df_cohort['Lapsed'] == 1].copy()
    terminated_detail_df = terminated_detail_df[['Cohort_Week_Index', 'Lapse_Week', 'Policy_Month', 'Termination_Reason'] + [col for col in policy_detail_cols if col in terminated_detail_df.columns and col not in ['policy_nbr', 'issue_date', 'term_date']]].copy() # Added 'Termination_Reason' for detail table
    
    # Return the modified tuple
    # MODIFIED: reasons_df is replaced by terminated_detail_df
    return retention_rate, total_lapsed_policies, cohort_sizes, cohort_dates_map, mix_df, terminated_detail_df, avg_months_held_df, initial_mix_df

# --- 3. NEW AGGREGATION AND VISUALIZATION FUNCTIONS (UNCHANGED) ---
def aggregate_average_months_held(df_avg: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """ Aggregates the persistency across multiple cohorts within a time frame. """
    
    df_filtered = df_avg[
        (df_avg['Cohort_Date'] >= start_date) & 
        (df_avg['Cohort_Date'] <= end_date)
    ].copy()
    
    if df_filtered.empty:
        return pd.DataFrame()

    # The aggregation needs to be weighted by Policy Count (Policies)
    df_filtered['Weighted_Months'] = df_filtered['Avg_Months_Held'] * df_filtered['Policies']
    
    # Only aggregate by segments (Underwriting Class, Gender, Tobacco)
    summary_df = df_filtered[df_filtered['Segment'] != 'Cohort Overall'].groupby(
        ['Segment', 'Category']
    ).agg(
        Total_Weighted_Months=('Weighted_Months', 'sum'),
        Total_Policies=('Policies', 'sum')
    ).reset_index()
    
    summary_df['Avg_Months_Held'] = summary_df['Total_Weighted_Months'] / summary_df['Total_Policies']
    summary_df.rename(columns={'Total_Policies': 'Policies'}, inplace=True)
    
    return summary_df[['Segment', 'Category', 'Avg_Months_Held', 'Policies']]


def plot_segment_avg_months(df_avg: pd.DataFrame, segment: str, title: str, key: str):
    """ Generates a bar chart for average months held by segment (Used for Summary Tab). """
    segment_df = df_avg[df_avg['Segment'] == segment].sort_values('Avg_Months_Held', ascending=False)
    
    if segment_df.empty:
        st.warning(f"No data to display for {segment} in the selected timeframe.")
        return

    fig = px.bar(
        segment_df,
        x='Avg_Months_Held',
        y='Category',
        orientation='h',
        title=title,
        text=segment_df['Avg_Months_Held'].round(2).astype(str) + ' mos',
        labels={'Avg_Months_Held': 'Average Months Held', 'Category': segment},
        color='Policies', # Use policies as color intensity
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      xaxis_title="Average Months Held (Persistency)",
                      height=max(400, 30 * len(segment_df)))
    st.plotly_chart(fig, use_container_width=True, key=key)


def plot_segment_avg_months_single_cohort(df_avg: pd.DataFrame, cohort_index: int, segment: str, cohort_map: dict):
    """ Generates a bar chart for average months held for a single cohort, segmented by attribute. """
    
    df_cohort = df_avg[
        (df_avg['Cohort_Index'] == cohort_index) & 
        (df_avg['Segment'] == segment)
    ].sort_values('Avg_Months_Held', ascending=False)
    
    if df_cohort.empty:
        st.warning(f"No data to display for {segment} in Cohort {cohort_index}.")
        return

    # Extract overall average for the title
    overall_avg = df_avg[
        (df_avg['Cohort_Index'] == cohort_index) & 
        (df_avg['Segment'] == 'Cohort Overall')
    ]['Avg_Months_Held'].iloc[0] if not df_avg.empty else 0.0

    st.subheader(f"Avg Months Held by {segment} in {cohort_map[cohort_index]}")
    st.markdown(f"**Cohort Overall Average:** `{overall_avg:.2f} months` (Policies: {df_cohort['Policies'].sum():,})")

    fig = px.bar(
        df_cohort,
        x='Avg_Months_Held',
        y='Category',
        orientation='h',
        text=df_cohort['Avg_Months_Held'].round(2).astype(str) + ' mos',
        labels={'Avg_Months_Held': 'Average Months Held', 'Category': segment},
        color='Policies',
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      xaxis_title="Average Months Held (Persistency)",
                      height=max(400, 30 * len(df_cohort)))
    # Added unique key
    st.plotly_chart(fig, use_container_width=True, key=f'persistency_bar_{cohort_index}_{segment.replace(" ", "_")}')

def plot_initial_mix_trends(df_mix: pd.DataFrame, segment: str, cohort_dates_df: pd.DataFrame):
    """ Generates a stacked area chart showing initial mix trends over time. """
    
    df_segment = df_mix[df_mix['Segment'] == segment]
    
    if df_segment.empty:
        st.warning(f"No initial mix data available for {segment}.")
        return
        
    df_segment = pd.merge(df_segment, cohort_dates_df.reset_index(), on='Cohort_Week_Index')
    
    fig = px.area(
        df_segment,
        x='Cohort_Start_Date',
        y='Proportion',
        color='Category',
        line_group='Category',
        title=f'Initial Policy Mix by {segment} Over Time',
        labels={'Cohort_Start_Date': 'Cohort Week Start Date', 'Proportion': 'Initial Policy Proportion (%)', 'Category': segment},
        groupnorm='percent' # Stacks to 100%
    )
    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Initial Policy Proportion (%)",
        hovermode="x unified"
    )
    # Added unique key
    st.plotly_chart(fig, use_container_width=True, key=f'mix_trend_{segment.replace(" ", "_")}')

# --- NEW FUNCTION FOR CONDITIONAL FORMATTING ---
def color_retention_matrix(df: pd.DataFrame):
    """ Applies green-to-red conditional formatting to the retention matrix. """
    
    # Define the custom 10-shade colormap from Green (high retention) to Red (low retention)
    # Note: Green is first (0-10%) and Red is last (90-100%) for retention rate
    colors = ["#f65420", "#f1733e", "#eb915d", "#e6b07c", "#e0cf9b", "#c9e29f", "#b6d58e", "#a3c87e", "#8fbc6d", "#6aa84f"]
    cmap = LinearSegmentedColormap.from_list("RedToGreen", colors, N=10)
    
    # Apply the background gradient
    return df.style.background_gradient(
        cmap=cmap, 
        vmin=0, 
        vmax=100, 
        subset=pd.IndexSlice[:, df.columns[0]:] # Apply to all columns except index
    # MODIFICATION: REMOVED 'null_color' ARGUMENT TO FIX CRITICAL ERROR
    ).format("{:.2f}").highlight_null()

# --------------------------------------------------------------------------------------------------
# --- Streamlit App Layout ---
# --------------------------------------------------------------------------------------------------


st.set_page_config(layout="wide", page_title="Weekly Cohort Retention Analysis")


st.title("🚀 Local Cohort Retention and Termination Analysis") # MODIFIED: Removed 'Reason'
st.markdown(f"Data is automatically loaded from the configured local directory: `{BASE_PATH}`")

# --- Initial Data Loading ---
try:
    # MODIFICATION: Changed how initial_df and earliest_date are loaded
    # Compute CSV mtime and pass it to loader so Streamlit cache invalidates when the file changes
    csv_path_for_cache = os.path.join(os.path.dirname(__file__), 'randompolicydata.csv')
    csv_mtime = os.path.getmtime(csv_path_for_cache) if os.path.exists(csv_path_for_cache) else None
    with st.spinner("Scanning directory and merging files (This may take a moment)..."):
        initial_df, earliest_date = load_and_merge_data(POLICY_FILE_PATTERN, NSF_FILE_PATTERN, CANCELLED_FILE_PATTERN, cache_buster=csv_mtime)

    if initial_df.empty or pd.isna(earliest_date):
        st.error(f"❌ Could not load or merge data, or no valid 'app_recvd_date' found. Please check the `BASE_PATH` ({BASE_PATH}) and ensure files are present.")
        st.stop()

    st.success(f"✅ Data loaded successfully. Total policies merged: {len(initial_df):,}")

    # --- Sidebar for Data Filtering ---
    st.sidebar.header("1. Analysis Configuration")
    
    # --- START MODIFICATION FOR MIN DATE CONSTRAINT ---
    MIN_ALLOWED_DATE = pd.to_datetime('2024-01-01').date()
    
    latest_available_date = initial_df['app_recvd_date'].max()
    data_earliest_date = earliest_date.date()
    
    # The default should be the earliest available date in the data, but no earlier than MIN_ALLOWED_DATE
    default_cohort_start = max(data_earliest_date, MIN_ALLOWED_DATE)
    
    cohort_start_date_input = st.sidebar.date_input(
        "Cohort Start Date (App Recvd Filter)", 
        value=default_cohort_start,
        min_value=MIN_ALLOWED_DATE, # Set minimum to Jan 1, 2024
        max_value=latest_available_date.date(),
        help=f"Select a cohort start date. Restricted to be no earlier than {MIN_ALLOWED_DATE.strftime('%Y-%m-%d')}."
    )
    cohort_start_date = pd.to_datetime(cohort_start_date_input)
    # --- END MODIFICATION FOR MIN DATE CONSTRAINT ---

    analysis_end_date_input = st.sidebar.date_input("Analysis Cut-off Date", value=ANALYSIS_END_DATE)
    ANALYSIS_END_DATE = pd.to_datetime(analysis_end_date_input)

    st.sidebar.header("2. Data Filtering")

    def get_unique_values(df, col_name):
        if col_name in df.columns:
            # Added a robust way to get unique, non-null values
            return df[col_name].astype(str).str.strip().replace('nan', pd.NA).dropna().unique().tolist()
        return []

    # Get ALL unique plan codes from the initially loaded data
    all_plan_options = get_unique_values(initial_df, 'plan_code')
    luminary_plan_codes = MAPPING_PLAN_CODES
    other_plan_codes = [p for p in all_plan_options if p not in luminary_plan_codes]

    # --- Plan Code Filter Group Selection ---
    st.sidebar.subheader("Filter by Plan Code Group")
    plan_group_selection = st.sidebar.radio(
        "Plan Code Group:",
        ('All', 'Luminary', 'Other', 'Custom Select'),
        index=0, # Default to 'All'
        key='plan_code_group_select'
    )

    selected_plans_for_analysis = []
    
    if plan_group_selection == 'All':
        selected_plans_for_analysis = all_plan_options
        st.sidebar.info(f"Filtering for **{len(selected_plans_for_analysis)}** total unique plan codes.")
    elif plan_group_selection == 'Luminary':
        selected_plans_for_analysis = luminary_plan_codes
        st.sidebar.info(f"Filtering for **{len(selected_plans_for_analysis)}** Luminary plan codes.")
    elif plan_group_selection == 'Other':
        selected_plans_for_analysis = other_plan_codes
        st.sidebar.info(f"Filtering for **{len(other_plan_codes)}** 'Other' plan codes.")
    elif plan_group_selection == 'Custom Select':
        st.sidebar.markdown("---")
        st.sidebar.subheader("Custom Plan Code Selection")
        # Multiselect for ALL plan codes when 'Custom Select' is chosen
        selected_plans_for_analysis = st.sidebar.multiselect(
            "Select specific Plan Codes:", 
            options=all_plan_options, 
            default=all_plan_options,
            key='plan_code_multiselect_custom'
        )
        st.sidebar.info(f"Custom selection: **{len(selected_plans_for_analysis)}** plan codes selected.")
    
    # Use the result of the group/custom selection as the filter argument
    selected_plans = selected_plans_for_analysis
    # --- END Plan Code Filter Group Selection ---

    selected_tobacco = []
    if 'tobacco' in initial_df.columns:
        tobacco_options = get_unique_values(initial_df, 'tobacco')
        selected_tobacco = st.sidebar.multiselect("Filter by Tobacco Status", options=tobacco_options, default=tobacco_options)

    selected_gender = []
    if 'gender' in initial_df.columns:
        gender_options = get_unique_values(initial_df, 'gender')
        selected_gender = st.sidebar.multiselect("Filter by Gender", options=gender_options, default=gender_options)
    
    # --- Case Status Filter ---
    case_status_options = ['Active', 'Inactive']
    selected_case_status = st.sidebar.multiselect(
        "Filter by Case Status", 
        options=case_status_options, 
        default=case_status_options, 
        help="Active: Not lapsed as of Analysis Cut-off Date. Inactive: Lapsed as of Analysis Cut-off Date."
    )
    # --------------------------

    # --- Run Cohort Analysis ---
    with st.spinner("Performing Cohort Analysis..."):
        # MODIFIED: removed reasons_df from the tuple
        retention_matrix, total_lapsed_policies, cohort_sizes, cohort_dates_df, mix_df, terminated_detail_df, avg_months_held_df, initial_mix_df = perform_cohort_analysis(
            initial_df.copy(), 
            cohort_start_date,
            selected_plans, # Pass the resolved list of plan codes
            selected_tobacco,
            selected_gender,
            selected_case_status
        )

    if retention_matrix.empty:
        st.warning("No policies found matching the selected filters and cohort start date.")
        st.stop()
    
    st.header("Results Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Cohort Policies", value=f"{cohort_sizes.sum():,}")
    with col2:
        st.metric("Total Policies Lapsed", value=f"{total_lapsed_policies:,}")

    cohort_map = {index: f"Cohort {index} (Week of {date.strftime('%Y-%m-%d')})" for index, date in cohort_dates_df['Cohort_Start_Date'].items()}


    # ----------------------------------------------------------------------------------
    # --- MAIN CONTENT TABS ---
    # ----------------------------------------------------------------------------------
    # MODIFIED: Removed 'Cross-Cohort Trends' and 'Cohort Comparison' tabs
    cohort_tab, persistency_summary_tab, persistency_deep_dive_tab, historical_mixes_tab = st.tabs([
        "📊 Cohort Deep Dive", 
        "💰 Cohort Persistency Summary", 
        "🔎 Single Cohort Persistency Deep Dive", 
        "📈 Historical Mixes"
    ])

    # ----------------------------------------------------------------------------------
    # --- TAB 1: Selected Cohort Deep Dive (Retention and Mix) ---
    # ----------------------------------------------------------------------------------
    with cohort_tab:
        st.header("Cohort Retention Matrix (%)")
        
        # MODIFICATION: Apply conditional formatting
        st.dataframe(color_retention_matrix(retention_matrix), use_container_width=True, key='retention_matrix_df')
        st.markdown("---")
        
        # Retention Plot
        plot_df = retention_matrix.reset_index().melt(id_vars='Cohort_Week_Index', var_name='Tenure_Week', value_name='Retention_Rate')
        plot_df['Tenure_Week_Num'] = plot_df['Tenure_Week'].str.split('_').str[1].astype(int)

        st.header("Retention Rate Over Tenure (All Cohorts)")
        fig = px.line(plot_df, x='Tenure_Week_Num', y='Retention_Rate', color='Cohort_Week_Index', title='Retention Rate by Cohort', markers=True, color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(xaxis_title="Tenure Week", yaxis_title="Retention Rate (%)", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, key='retention_rate_plot_all')
        st.markdown("---")

        # Cohort Deep Dive & Mix Analysis
        st.header("Single Cohort Deep Dive: Retention Curve & Mix")
        selected_cohort_index = list(cohort_map.keys())[0] # Default selection
        
        col_sel, col_group = st.columns([0.6, 0.4])
        with col_sel:
            selected_cohort_index = st.selectbox("Select a Cohort:", options=list(cohort_map.keys()), format_func=lambda x: cohort_map[x], key='deep_dive_cohort')
        with col_group:
            group_by_col = st.radio("Group Mix Chart By:", ('Underwriting Class', 'Gender', 'Tobacco'), key='deep_dive_group')
        group_col_name = {'Underwriting Class': 'underwriting_class', 'Gender': 'gender', 'Tobacco': 'tobacco'}[group_by_col]
        
        single_mix_df = mix_df[mix_df['Cohort_Week_Index'] == selected_cohort_index].copy()
        max_tenure_cohort = int(single_mix_df['Tenure_Weeks'].max()) if not single_mix_df.empty else 0
        mix_data = []
        single_retention_df = plot_df[plot_df['Cohort_Week_Index'] == selected_cohort_index].dropna(subset=['Retention_Rate'])

        for week_num in range(1, max_tenure_cohort + 1):
            policies_active_at_end_of_week = single_mix_df[(single_mix_df['Lapsed'] == 0) | (single_mix_df['Lapse_Week'] > week_num)]
            if policies_active_at_end_of_week.empty: break
            
            # FIX: Only attempt to group by the column if it exists in the filtered DataFrame
            if group_col_name in policies_active_at_end_of_week.columns:
                group_counts = policies_active_at_end_of_week.groupby(group_col_name)['Lapsed'].count().reset_index(name='Active_Policies')
                total_active = group_counts['Active_Policies'].sum()
                if total_active > 0:
                    group_counts['Proportion_Active'] = (group_counts['Active_Policies'] / total_active) * 100
                    group_counts['Tenure_Week_Num'] = week_num
                    mix_data.append(group_counts)
            else:
                # If the column is missing after filtering (e.g., all policies had same value), skip the mix calculation for this segment
                continue

        
        if mix_data and not single_retention_df.empty:
            final_mix_df = pd.concat(mix_data)
            
            # Plot 1: Retention Curve & Mix (Dual Axis)
            fig_combined = make_subplots(specs=[[{"secondary_y": True}]])
            for group in final_mix_df[group_col_name].unique():
                group_data = final_mix_df[final_mix_df[group_col_name] == group]
                fig_combined.add_trace(go.Scatter(x=group_data['Tenure_Week_Num'], y=group_data['Proportion_Active'], name=f"Mix: {group}", mode='lines', fill='tonexty', line=dict(width=0), stackgroup='one', hovertemplate=f"Week: %{{x}}<br>{group_by_col}: {group}<br>Mix: %{{y:.1f}}%<extra></extra>"), secondary_y=False)
            fig_combined.add_trace(go.Scatter(x=single_retention_df['Tenure_Week_Num'], y=single_retention_df['Retention_Rate'], name='Overall Retention Rate', mode='lines+markers', line=dict(color='black', width=3), marker=dict(symbol='circle', size=6), hovertemplate="Week: %{x}<br>Retention: %{y:.2f}%<extra></extra>"), secondary_y=True)
            fig_combined.update_layout(title_text=f"Retention Curve (Line) & Policy Mix (Area) for {cohort_map[selected_cohort_index]}", height=500, hovermode="x unified", legend_title_text="Legend")
            fig_combined.update_xaxes(title_text="Tenure Week")
            fig_combined.update_yaxes(title_text="Active Policy Mix (%)", secondary_y=False, range=[0, 100], showgrid=False)
            fig_combined.update_yaxes(title_text="Overall Retention Rate (%)", secondary_y=True, range=[0, 100], showgrid=True)
            st.plotly_chart(fig_combined, use_container_width=True, key='retention_curve_mix_plot')
            
            st.subheader(f"Absolute Count of Active Policies by {group_by_col}")
            fig_bars = px.bar(final_mix_df, x='Tenure_Week_Num', y='Active_Policies', color=group_col_name, barmode='group', title=f'Active Policy Counts for {cohort_map[selected_cohort_index]}')
            fig_bars.update_layout(xaxis_title="Tenure Week", yaxis_title="Count of Active Policies", legend_title_text=group_col_name, height=400)
            st.plotly_chart(fig_bars, use_container_width=True, key='active_policy_count_plot')
            st.markdown("---")
            
            # LAPSE REASON ANALYSIS REMOVED
            # st.header("Lapse Reason Breakdown Over Tenure (Single Cohort)") ... (Removed all charts related to reason analysis)
            
            # TERMINATED POLICY DETAIL TABLE (Simplified to remove heavy 'reasons' focus)
            st.header("📝 Detailed List of Terminated Policies (Reference Data)")
            # MODIFIED: Use terminated_detail_df instead of reasons_df
            detail_table_df = terminated_detail_df[terminated_detail_df['Cohort_Week_Index'] == selected_cohort_index].copy()

            # Fetch the policies again from the main cohort DF to get all columns (issue/term dates, policy_nbr)
            detail_policy_nrs = detail_table_df['policy_nbr'].unique()
            # Re-filter the main df_cohort which has all policy details including dates
            full_detail_df = initial_df[initial_df['policy_nbr'].isin(detail_policy_nrs)].copy()
            
            # Merge the simplified terminated_detail_df back to get Cohort/Lapse Week/Reason
            merge_cols = ['policy_nbr', 'term_date', 'issue_date', 'plan_code', 'tobacco', 'gender', 'Termination_Reason', 'Lapse_Week', 'Policy_Month']
            detail_table_df = full_detail_df[merge_cols].dropna(subset=['term_date']).copy()
            
            # FIX: Check if date columns exist and are datetime before formatting
            if 'term_date' in detail_table_df.columns and pd.api.types.is_datetime64_any_dtype(detail_table_df['term_date']):
                detail_table_df['Terminated_Date'] = detail_table_df['term_date'].dt.strftime('%Y-%m-%d').fillna('N/A')
            else:
                detail_table_df['Terminated_Date'] = 'N/A'
                
            if 'issue_date' in detail_table_df.columns and pd.api.types.is_datetime64_any_dtype(detail_table_df['issue_date']):
                detail_table_df['Issue_Date'] = detail_table_df['issue_date'].dt.strftime('%Y-%m-%d').fillna('N/A')
            else:
                detail_table_df['Issue_Date'] = 'N/A'
                
            # MODIFIED: Retained 'Termination_Reason' for the table as it's a core attribute of the terminated policy, even without a chart.
            display_cols = ['policy_nbr', 'Terminated_Date', 'Issue_Date', 'Lapse_Week', 
                            'Termination_Reason', 'plan_code', 'gender', 'tobacco']
            final_display_cols = [col for col in display_cols if col in detail_table_df.columns]
            
            st.dataframe(detail_table_df[final_display_cols].set_index('policy_nbr'), use_container_width=True, key='terminated_policies_detail')
            st.markdown("---")
            
            # --- REFERENCE DATA FOR SINGLE COHORT ---
            st.subheader("Reference Data: Active Policy Mix Over Tenure")
            final_mix_df_display = final_mix_df.copy()
            final_mix_df_display.rename(columns={'Tenure_Week_Num': 'Tenure Week', group_col_name: 'Category', 'Active_Policies': 'Active Policy Count', 'Proportion_Active': 'Mix (%)'}, inplace=True)
            st.dataframe(final_mix_df_display[['Tenure Week', 'Category', 'Active Policy Count', 'Mix (%)']].style.format({'Mix (%)': '{:.2f}'}), use_container_width=True, key='ref_mix_data_deep_dive')

            # REMOVED: Reference Data: Lapse Reason Count and Proportion
            
        else:
            st.info("No active policies found for the selected cohort or filters to generate the mix chart.")
            

    # ----------------------------------------------------------------------------------
    # --- TAB 2: Cross-Cohort Trends (REMOVED: Lapse Reason Trends) ---
    # ----------------------------------------------------------------------------------
    # with cross_cohort_tab: (ENTIRE TAB REMOVED)


    # ----------------------------------------------------------------------------------
    # --- TAB 3: Cohort Comparison (REMOVED: Lapse Reason Comparison) ---
    # ----------------------------------------------------------------------------------
    # with cohort_comparison_tab: (ENTIRE TAB REMOVED)


    # ----------------------------------------------------------------------------------
    # --- TAB 4: Cohort Persistency Summary (Now the 2nd visible tab) ---
    # ----------------------------------------------------------------------------------
    with persistency_summary_tab:
        st.header("💰 Cohort Persistency Summary (Aggregated Timeframe)")
        st.info("This view aggregates the Average Months Held for all cohorts that started within the selected time period, segmented by policy attributes.")

        if avg_months_held_df.empty:
            st.warning("No data available to calculate average months held with current filters.")
        else:
            min_date = cohort_dates_df['Cohort_Start_Date'].min()
            max_date = cohort_dates_df['Cohort_Start_Date'].max()
            
            col_date1, col_date2 = st.columns(2)

            with col_date1:
                start_date_filter = st.date_input("Filter Cohort Start Date (From)", value=min_date, min_value=min_date, max_value=max_date, key='persistency_summary_start')
            with col_date2:
                end_date_filter = st.date_input("Filter Cohort Start Date (To)", value=max_date, min_value=min_date, max_value=max_date, key='persistency_summary_end')
            
            summary_data = aggregate_average_months_held(avg_months_held_df, pd.to_datetime(start_date_filter), pd.to_datetime(end_date_filter))
            
            if summary_data.empty:
                st.warning("No cohorts were found within the selected date range.")
            else:
                
                # Overall Average Calculation for the filtered group
                overall_summary_df = avg_months_held_df[
                    (avg_months_held_df['Cohort_Date'] >= pd.to_datetime(start_date_filter)) & 
                    (avg_months_held_df['Cohort_Date'] <= pd.to_datetime(end_date_filter)) &
                    (avg_months_held_df['Segment'] == 'Cohort Overall')
                ]
                
                if not overall_summary_df.empty and overall_summary_df['Policies'].sum() > 0:
                    overall_avg_months = (overall_summary_df['Avg_Months_Held'] * overall_summary_df['Policies']).sum() / overall_summary_df['Policies'].sum()
                    st.metric("Overall Average Months Held (Filtered Cohorts)", f"{overall_avg_months:.2f} months", delta=None)
                else:
                    st.metric("Overall Average Months Held (Filtered Cohorts)", "N/A", delta=None)


                st.markdown("---")

                # Plotting the aggregated segments
                st.subheader("Persistency by Underwriting Class")
                plot_segment_avg_months(summary_data, 'Underwriting Class', 'Aggregated Avg Months Held by Underwriting Class', 'persistency_uw_summary')

                st.subheader("Persistency by Gender")
                plot_segment_avg_months(summary_data, 'Gender', 'Aggregated Avg Months Held by Gender', 'persistency_gender_summary')

                st.subheader("Persistency by Tobacco Status")
                plot_segment_avg_months(summary_data, 'Tobacco', 'Aggregated Avg Months Held by Tobacco Status', 'persistency_tobacco_summary')

                st.markdown("---")
                
                # --- REFERENCE DATA FOR PERSISTENCY SUMMARY ---
                st.subheader("Reference Data: Aggregated Persistency Data Table")
                st.dataframe(summary_data.round({'Avg_Months_Held': 2}).rename(columns={'Policies': 'Policy Count', 'Avg_Months_Held': 'Avg Months Held'}), use_container_width=True, key='persistency_aggregated_table')


    # ----------------------------------------------------------------------------------
    # --- TAB 5: Single Cohort Persistency Deep Dive (Now the 3rd visible tab) ---
    # ----------------------------------------------------------------------------------
    with persistency_deep_dive_tab:
        st.header("🔎 Single Cohort Persistency Deep Dive")
        st.info("This view isolates one selected cohort and analyzes the average months held for its sub-segments.")

        if avg_months_held_df.empty:
            st.warning("No data available to calculate average months held with current filters.")
        else:
            col_sel, col_group = st.columns([0.4, 0.6])
            
            with col_sel:
                selected_persistency_cohort_index = st.selectbox(
                    "Select Cohort for Analysis:", 
                    options=list(cohort_map.keys()), 
                    format_func=lambda x: cohort_map[x], 
                    key='persistency_cohort_select_deep_dive'
                )
                
            with col_group:
                segment_for_persistency = st.radio(
                    "Segment Persistency By:", 
                    ('Underwriting Class', 'Gender', 'Tobacco'),
                    key='persistency_group_radio_deep_dive',
                    horizontal=True
                )
                
            st.markdown("---")
            
            plot_segment_avg_months_single_cohort(
                avg_months_held_df, 
                selected_persistency_cohort_index, 
                segment_for_persistency, 
                cohort_map
            )
            
            st.markdown("---")
            
            # --- REFERENCE DATA FOR SINGLE COHORT PERSISTENCY ---
            st.subheader("Reference Data: Segmented Persistency Metrics (Selected Cohort)")
            single_cohort_persistency_data = avg_months_held_df[
                   (avg_months_held_df['Cohort_Index'] == selected_persistency_cohort_index) & 
                   (avg_months_held_df['Segment'] != 'Cohort Overall')
            ].copy()
            single_cohort_persistency_data = single_cohort_persistency_data[['Segment', 'Category', 'Avg_Months_Held', 'Policies']].rename(
                columns={'Policies': 'Policy Count', 'Avg_Months_Held': 'Avg Months Held'}
            )
            st.dataframe(single_cohort_persistency_data.style.format({"Avg Months Held": "{:.2f}"}), use_container_width=True, key='ref_persistency_single_cohort_data')

            st.subheader("Reference Data: Overall Cohort Persistency Summary")
            
            # Create a table showing the overall cohort persistency
            cohort_summary_table = avg_months_held_df[
                avg_months_held_df['Segment'] == 'Cohort Overall'
            ].copy().sort_values('Cohort_Index')
            
            cohort_summary_table['Cohort'] = 'Cohort ' + cohort_summary_table['Cohort_Index'].astype(str) + ' (' + cohort_summary_table['Cohort_Date'].dt.strftime('%Y-%m-%d') + ')'
            
            final_table = cohort_summary_table[['Cohort', 'Avg_Months_Held', 'Policies']].rename(
                columns={'Avg_Months_Held': 'Avg Months Held', 'Policies': 'Policy Count'}
            ).set_index('Cohort')
            
            st.dataframe(final_table.style.format({"Avg Months Held": "{:.2f}"}), use_container_width=True, key='persistency_summary_table_all')

    # ----------------------------------------------------------------------------------
    # --- TAB 6: Historical Mixes (Now the 4th visible tab) ---
    # ----------------------------------------------------------------------------------
    with historical_mixes_tab:
        st.header("📈 Initial Policy Mix Trends by Cohort")
        st.info("This view shows how the demographic and underwriting mix of your new business has changed over time for the policies included in the analysis.")
        
        if initial_mix_df.empty:
            st.warning("No initial mix data available.")
        else:
            
            st.subheader("1. Underwriting Class Mix Trend")
            plot_initial_mix_trends(initial_mix_df, 'Underwriting Class', cohort_dates_df)

            st.markdown("---")
            
            st.subheader("2. Gender Mix Trend")
            plot_initial_mix_trends(initial_mix_df, 'Gender', cohort_dates_df)
            
            st.markdown("---")

            st.subheader("3. Tobacco Status Mix Trend")
            plot_initial_mix_trends(initial_mix_df, 'Tobacco', cohort_dates_df)
            
            st.markdown("---")
            
            # --- REFERENCE DATA FOR HISTORICAL MIXES ---
            st.subheader("Reference Data: Initial Mix Data Table")
            mix_table = pd.merge(initial_mix_df, cohort_dates_df.reset_index(), on='Cohort_Week_Index')
            mix_table['Cohort Date'] = mix_table['Cohort_Start_Date'].dt.strftime('%Y-%m-%d')
            mix_table = mix_table[['Cohort Date', 'Cohort_Week_Index', 'Segment', 'Category', 'Policy_Count', 'Proportion']]
            mix_table.rename(columns={'Cohort_Week_Index': 'Cohort Index', 'Policy_Count': 'Count', 'Proportion': 'Proportion (%)'}, inplace=True)
            
            st.dataframe(mix_table.round({'Proportion (%)': 2}), use_container_width=True, hide_index=True, key='initial_mix_data_table')


except Exception as e:
    # Ensure the error is captured and displayed for debugging
    st.error(f"A critical error occurred during file processing or analysis: {e}")
    st.info("Double-check that the `BASE_PATH` is correct and all required files (`Policies_DBGA_*.csv`, `NSF_Activity_DBGA_*.csv`, `Cancelled_*.csv`) are present and properly formatted.")
