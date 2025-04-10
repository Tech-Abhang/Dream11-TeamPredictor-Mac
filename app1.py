import pandas as pd
import numpy as np
import pulp
from pycaret.regression import load_model, predict_model
import os
import warnings
import sys
from typing import Tuple, Optional, Dict
from functools import partial, lru_cache
from pathlib import Path

warnings.filterwarnings('ignore')

# Constants
MASTER_FILE_PATH = 'TEST/main_file_updated.csv'
MODEL_PATH = 'TEST/ml_avg_predictor_model.pkl'
INPUT_EXCEL_PATH = r"C:\Users\rajen\OneDrive\Desktop\Downloads\SquadPlayerNames_IndianT20League.xlsx"
DOWNLOADS_PATH = str(Path.home() / "Downloads")  # Gets user's Downloads folder
MAX_CREDITS = 100
TEAM_SIZE = 11
NAME_CORRECTIONS: Dict[str, str] = {}  # Add name mappings as needed

# Feature lists
CAREER_FEATURES = [
    'Innings', 'Runs', 'Balls Faced', 'Fours', 'Sixes', 'Batting Average',
    'Strike Rate', 'Boundary_Percentage', 'Innings_X', 'Balls Bowled',
    'Runs_Conceded', 'Wickets', 'Bowling Average', 'Economy Rate',
    'Bowling Strike_rate'
]
MATCH_FEATURES = ['Credits', 'Player Type', 'Team', 'Weighted_Score']

# Data Loading with Python's built-in caching
@lru_cache(maxsize=1)
def load_master_data(file_path: str = MASTER_FILE_PATH) -> Optional[pd.DataFrame]:
    try:
        if not os.path.exists(file_path):
            print(f"❌ Master file not found at: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.replace('  ', ' ').str.strip()
        df['Player Name Clean'] = df['Player Name'].str.strip().replace(NAME_CORRECTIONS)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        avg_cols = ['Dream11_Avg', 'Dream_11 round 1', 'Dream_11_2024_avg']
        if all(col in df.columns for col in avg_cols):
            for col in avg_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['Weighted_Score'] = (
                0.60 * df['Dream11_Avg'] +
                0.10 * df['Dream_11_2024_avg'] +
                0.30 * df['Dream_11 round 1']
            )
        else:
            df['Weighted_Score'] = 0
        return df
    except Exception as e:
        print(f"❌ Failed to load master data: {e}")
        return None

@lru_cache(maxsize=1)
def load_ml_model(model_path: str = MODEL_PATH) -> Optional:
    try:
        if not os.path.exists(model_path):
            print(f"❌ Model not found at: {model_path}")
            print(f"Please run TRAIN1.py first to create the model")
            return None
            
        return load_model(model_path)
    except Exception as e:
        print(f"❌ Failed to load ML model: {e}")
        return None

# Data Preparation
def prepare_data_for_prediction(
    match_players_df: pd.DataFrame,
    master_df: pd.DataFrame
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if master_df is None or not all(feat in master_df.columns for feat in CAREER_FEATURES):
        return None, None

    required_cols = ['Player Name', 'Credits', 'Player Type', 'Team']
    if not all(col in match_players_df.columns for col in required_cols):
        return None, None

    df = match_players_df.copy()
    df['Player Name Clean'] = df['Player Name'].str.strip().replace(NAME_CORRECTIONS)
    df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce').fillna(0)
    
    # Filter playing players if available
    if 'IsPlaying' in df.columns:
        playing_mask = df['IsPlaying'].str.upper().fillna('') == 'PLAYING'
        df = df[playing_mask] if playing_mask.any() else df
    
    # Handle lineup order
    df['lineupOrder'] = pd.to_numeric(
        df.get('lineupOrder', 99), errors='coerce').fillna(99)

    # Merge with master data
    master_subset = master_df[['Player Name Clean'] + CAREER_FEATURES + ['Weighted_Score']]
    merged_df = df.merge(master_subset, on='Player Name Clean', how='left')

    # Impute missing values
    missing_players = merged_df[merged_df[CAREER_FEATURES].isnull().any(axis=1)]['Player Name'].tolist()
    if missing_players:
        print(f"Missing stats for players: {missing_players[:10]}{'...' if len(missing_players) > 10 else ''}")
        for col in CAREER_FEATURES:
            if col in merged_df and merged_df[col].isnull().any():
                merged_df[col].fillna(master_df[col].median() or 0, inplace=True)

    # Type enforcement
    for col in MATCH_FEATURES + CAREER_FEATURES:
        if col in merged_df:
            merged_df[col] = (pd.to_numeric(merged_df[col], errors='coerce').fillna(0) 
                            if col != 'Player Type' and col != 'Team' 
                            else merged_df[col].astype(str).fillna('Unknown'))

    features = merged_df[MATCH_FEATURES + CAREER_FEATURES]
    info = merged_df[['Player Name', 'Credits', 'Player Type', 'Team', 'lineupOrder', 'Weighted_Score']]
    return features, info

# Team Selection
def select_best_team(
    features: pd.DataFrame,
    player_info: pd.DataFrame,
    ml_model,
    max_credits: float = MAX_CREDITS
) -> pd.DataFrame:
    if features.empty or player_info.empty or ml_model is None:
        return pd.DataFrame()
    
    try:
        predictions = predict_model(ml_model, data=features)
        if 'prediction_label' not in predictions.columns:
            return pd.DataFrame()
        
        pred_scores = predictions['prediction_label'].clip(lower=0).rename('Predicted_Score')
        optimization_df = pd.concat([player_info, pred_scores], axis=1)
        optimization_df = optimization_df.dropna(subset=['Credits', 'Predicted_Score', 'Player Name', 'Team', 'Player Type'])
        
        return select_best_team_optimized_pulp(optimization_df, max_credits)
    except Exception as e:
        print(f"❌ Prediction/Optimization error: {e}")
        return pd.DataFrame()

def select_best_team_optimized_pulp(
    df: pd.DataFrame,
    max_credits: float,
    lineup_bonus: float = 0.15,
    form_weight: float = 0.3
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    # Prepare data for optimization
    players = df['Player Name'].tolist()
    data = {
        'credits': df.set_index('Player Name')['Credits'].to_dict(),
        'scores': df.set_index('Player Name')['Predicted_Score'].to_dict(),
        'roles': df.set_index('Player Name')['Player Type'].str.upper().to_dict(),
        'teams': df.set_index('Player Name')['Team'].to_dict(),
        'lineup': df.set_index('Player Name')['lineupOrder'].to_dict()
    }
    
    # Add weighted score if available for form consideration
    if 'Weighted_Score' in df.columns:
        data['form'] = df.set_index('Player Name')['Weighted_Score'].to_dict()
    else:
        data['form'] = {p: 0 for p in players}
    
    # Create improved score calculation with multiple factors
    adjusted_scores = {}
    for p in players:
        base_score = data['scores'][p]
        
        # Batting lineup position bonus (especially for top order)
        lineup_position = data['lineup'][p]
        batting_bonus = 0
        if 'BAT' in data['roles'][p] and lineup_position <= 5:
            # Stronger bonus for positions 1-3, diminishing after
            batting_bonus = lineup_bonus * (6 - lineup_position) / 5.0 * base_score
        
        # Form consideration
        form_bonus = data['form'][p] * form_weight
        
        # Role-specific adjustments
        role_bonus = 0
        if 'ALL' in data['roles'][p] or 'ROUNDER' in data['roles'][p]:
            # All-rounders have more opportunities to score
            role_bonus = 0.15 * base_score
        elif 'BOWL' in data['roles'][p] and lineup_position <= 8:
            # Bowling all-rounders (bowlers who can bat)
            role_bonus = 0.1 * base_score
        
        # Calculate final adjusted score
        adjusted_scores[p] = base_score + batting_bonus + form_bonus + role_bonus
    
    # Optimization problem
    prob = pulp.LpProblem("Dream11_Team", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("Player", players, cat='Binary')
    
    # Objective: Maximize adjusted scores
    prob += pulp.lpSum(adjusted_scores[p] * player_vars[p] for p in players)
    
    # Budget constraint
    prob += pulp.lpSum(data['credits'][p] * player_vars[p] for p in players) <= max_credits
    
    # Team size constraint
    prob += pulp.lpSum(player_vars[p] for p in players) == TEAM_SIZE

    # Role constraints with more flexibility
    role_constraints = {
        'WK': (1, 4, lambda r: 'WK' in r or 'WICKETKEEPER' in r),
        'BAT': (1, 6, lambda r: 'BAT' in r and 'WICKETKEEPER' not in r),  # More batsmen for balance
        'AR': (1, 6, lambda r: 'ALL' in r or 'ROUNDER' in r),
        'BOWL': (1, 6, lambda r: 'BOWL' in r)  # More bowlers for balance
    }
    
    for role, (min_count, max_count, condition) in role_constraints.items():
        prob += pulp.lpSum(player_vars[p] for p in players if condition(data['roles'][p])) >= min_count
        prob += pulp.lpSum(player_vars[p] for p in players if condition(data['roles'][p])) <= max_count

    # Team constraint - maximum players from one team
    for team in set(data['teams'].values()):
        if pd.notna(team):
            prob += pulp.lpSum(player_vars[p] for p in players if data['teams'][p] == team) <= 10
    
    # Balance constraint - ensure we have players from both teams
    team_list = list(set(data['teams'].values()))
    if len(team_list) >= 2:
        for team in team_list:
            if pd.notna(team):
                prob += pulp.lpSum(player_vars[p] for p in players if data['teams'][p] == team) >= 1
    
    # Ensure we have some top order batsmen
    top_order_batsmen = [p for p in players if data['lineup'][p] <= 4 and 
                         ('BAT' in data['roles'][p] or 'WK' in data['roles'][p] or 
                          'WICKETKEEPER' in data['roles'][p])]
    if top_order_batsmen:
        prob += pulp.lpSum(player_vars[p] for p in top_order_batsmen) >= 3
    
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=20)  # Set time limit for larger problems
    prob.solve(solver)
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        # If optimal solution not found, try relaxing some constraints
        print(f"Optimal solution not found. Using best available solution: {pulp.LpStatus[prob.status]}")
    
    selected_players = [p for p in players if player_vars[p].varValue > 0.5]
    result_df = df[df['Player Name'].isin(selected_players)].copy()
    
    # Add the score used for selection to the result
    result_df['Score_Used'] = result_df['Player Name'].map(lambda p: adjusted_scores[p])
    
    return result_df

# Captain/Vice-Captain Assignment
def assign_captain_vice_captain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'Predicted_Score' not in df.columns:
        return df
    
    # Make a copy to avoid modifying the original dataframe
    df_cvc = df.copy()
    
    # Create features for C/VC consideration
    df_cvc['CVC_Score'] = df_cvc['Predicted_Score']
    
    # Add bonuses based on player attributes and recent form
    
    # Role-specific adjustments (more nuanced than before)
    role_bonuses = {
        'BAT': {
            'top_order': lambda row: 0.2 if 'BAT' in row['Player Type'].upper() and row.get('lineupOrder', 99) <= 3 else 0,
            'aggressive': lambda row: 0.15 if 'BAT' in row['Player Type'].upper() and row.get('Strike Rate', 0) > 150 else 0
        },
        'BOWL': {
            'death_bowler': lambda row: 0.15 if 'BOWL' in row['Player Type'].upper() and row.get('Economy Rate', 0) < 8.5 else 0,
            'wicket_taker': lambda row: 0.2 if 'BOWL' in row['Player Type'].upper() and row.get('Bowling Strike_rate', 0) < 18 else 0
        },
        'ALL': {
            'dual_threat': lambda row: 0.25 if ('ALL' in row['Player Type'].upper() or 'ROUNDER' in row['Player Type'].upper()) else 0
        },
        'WK': {
            'batting_wk': lambda row: 0.15 if ('WK' in row['Player Type'].upper() or 'WICKETKEEPER' in row['Player Type'].upper()) and row.get('lineupOrder', 99) <= 4 else 0
        }
    }
    
    # Apply role-specific bonuses
    for role, bonuses in role_bonuses.items():
        for bonus_name, bonus_func in bonuses.items():
            df_cvc['CVC_Score'] += df_cvc.apply(bonus_func, axis=1) * df_cvc['Predicted_Score']
    
    # Form-based adjustments
    if 'Weighted_Score' in df_cvc.columns:
        # Give bonus for consistent performers
        df_cvc['CVC_Score'] += df_cvc['Weighted_Score'] * 0.1
    
    # Lineup order considerations - favor top order batsmen and all-rounders
    if 'lineupOrder' in df_cvc.columns:
        # Batting position bonus (decreasing as we go down the order)
        df_cvc['CVC_Score'] += df_cvc.apply(
            lambda row: max(0, (5 - min(row.get('lineupOrder', 5), 5)) / 10) * row['Predicted_Score'] 
            if 'BAT' in row['Player Type'].upper() or 'ALL' in row['Player Type'].upper() or 'ROUNDER' in row['Player Type'].upper()
            else 0, 
            axis=1
        )
    
    # Consider fantasy-specific stats if available
    if 'Dream_11 round 1' in df_cvc.columns:
        df_cvc['CVC_Score'] += df_cvc['Dream_11 round 1'] * 0.10
    
    # Venue/opposition considerations could be added here if data is available
    
    # Select C and VC based on final CVC_Score
    captain_vice = df_cvc.nlargest(2, 'CVC_Score')['Player Name'].tolist()
    
    # Apply C/VC to original dataframe
    df['C/VC'] = df['Player Name'].map(
        lambda x: 'C' if x == captain_vice[0] else 
                  'VC' if len(captain_vice) > 1 and x == captain_vice[1] else ''
    )
    
    return df

# Streamlit UI
def main():
    # Check if match number is provided
    if len(sys.argv) != 2:
        print("Usage: python TEST/app1.py <match_number>")
        print("Example: python TEST/app1.py 1")
        return
        
    try:
        match_number = int(sys.argv[1])
        sheet_name = f"Match_{match_number}"
    except ValueError:
        print("❌ Match number must be a valid integer")
        return
    
    print(f"Loading data for {sheet_name}...")
    
    # Load required data
    master_data = load_master_data()
    if master_data is None:
        print(f"Please ensure {MASTER_FILE_PATH} exists in the current directory")
        return
        
    ml_model = load_ml_model()
    if ml_model is None:
        print(f"Please run TRAIN1.py first to create the model at {MODEL_PATH}")
        return
    
    print("✅ Successfully loaded master data and model")
    
    try:
        # Read match data from Excel file with specified sheet
        print(f"Reading match data from sheet: {sheet_name}")
        if not os.path.exists(INPUT_EXCEL_PATH):
            print(f"❌ Input Excel file not found at: {INPUT_EXCEL_PATH}")
            return
            
        try:
            match_data = pd.read_excel(INPUT_EXCEL_PATH, sheet_name=sheet_name)
            print("✅ Successfully loaded match data")
        except ValueError:
            print(f"❌ Sheet '{sheet_name}' not found in the Excel file")
            # List available sheets
            available_sheets = pd.ExcelFile(INPUT_EXCEL_PATH).sheet_names
            print("\nAvailable sheets:")
            for sheet in available_sheets:
                print(f"- {sheet}")
            return
        
        features, info = prepare_data_for_prediction(match_data, master_data)
        
        if features is not None and info is not None:
            best_team = select_best_team(features, info, ml_model)
            if not best_team.empty:
                # Assign captain and vice-captain
                final_team = assign_captain_vice_captain(best_team)
                
                # Create output filename
                output_filename = f'predicted_team_match_{match_number}.csv'
                output_path = os.path.join(DOWNLOADS_PATH, output_filename)
                
                # Select and save only required columns to CSV
                output_columns = ['Player Name', 'Team', 'C/VC']
                final_team[output_columns].to_csv(output_path, index=False)
                print("\n" + "="*50)
                print("PREDICTION COMPLETE!")
                print("="*50)
                print(f"\nTeam file saved at:\n{output_path}")
                print("\n" + "="*50)
                
                # Print some basic stats
                print(f"\nTeam Summary:")
                print(f"Credits Used: {final_team['Credits'].sum():.1f} / 100")
                
                # Role distribution
                roles_count = final_team['Player Type'].str.upper().apply(
                    lambda x: 'WK' if 'WICKETKEEPER' in x or 'WK' in x else
                              'AR' if 'ALL' in x or 'ROUNDER' in x else
                              'BAT' if 'BAT' in x else 'BOWL'
                ).value_counts()
                
                print("\nRole Distribution:")
                for role, count in roles_count.items():
                    print(f"{role}: {count}")
                
                # Team distribution
                print("\nTeam Distribution:")
                for team, count in final_team['Team'].value_counts().items():
                    print(f"{team}: {count}")
                
                # Print captain and vice-captain
                captain = final_team[final_team['C/VC'] == 'C']['Player Name'].iloc[0]
                vice_captain = final_team[final_team['C/VC'] == 'VC']['Player Name'].iloc[0]
                print(f"\nCaptain: {captain}")
                print(f"Vice Captain: {vice_captain}")
                
            else:
                print("❌ No valid team could be selected")
        else:
            print("❌ Failed to prepare data for prediction")
    except Exception as e:
        print(f"❌ Error processing file: {e}")

if __name__ == "__main__":
    main()