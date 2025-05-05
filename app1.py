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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
MASTER_FILE_PATH = 'main_file_updated.csv'
MODEL_PATH = 'model'  # No extension
INPUT_EXCEL_PATH = os.path.expanduser('~/Downloads/SquadPlayerNames_IndianT20League.xlsx')
DOWNLOADS_PATH = str(Path.home() / "Downloads")  # Gets user's Downloads folder
MAX_CREDITS = 100
TEAM_SIZE = 11
NAME_CORRECTIONS: Dict[str, str] = {}  # Add name mappings as needed

# Feature lists
CAREER_FEATURES = [
    'Innings', 'Runs', 'Balls Faced', 'Fours', 'Sixes', 'Batting Average',
    'Strike Rate', 'Boundary_Percentage', 'Innings_X', 'Balls Bowled',
    'Runs_Conceded', 'Wickets', 'Bowling Average', 'Economy Rate',
    'Bowling Strike_rate', 'RECENT_FORM'
]
MATCH_FEATURES = ['Credits', 'Player Type', 'Team']
FORM_FEATURES = ['Dream11_Avg', 'Dream_11 rounds_Avg', 'Dream_11_2024_avg', 'Weighted_Score', 'RECENT_FORM']

# Data Loading with Caching
@lru_cache(maxsize=1)
def load_master_data(file_path: str = MASTER_FILE_PATH) -> Optional[pd.DataFrame]:
    """
    Load and preprocess the master dataset.
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"Master file not found at: {file_path}")
            return None
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.replace('  ', ' ').str.strip()
        df['Player Name Clean'] = df['Player Name'].str.strip().replace(NAME_CORRECTIONS)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Handle form fields
        form_cols = [col for col in FORM_FEATURES if col in df.columns]
        for col in form_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Calculate Weighted_Score if missing
        if 'Weighted_Score' not in df.columns and 'Dream11_Avg' in df.columns:
            weights = [('Dream11_Avg', 0.5), ('Dream_11_2024_avg', 0.2), ('Dream_11 rounds_Avg', 0.3)]
            weights = [(col, w / sum(w for _, w in weights)) for col, w in weights if col in df.columns]
            df['Weighted_Score'] = sum(df[col] * w for col, w in weights) if weights else 0

        # Add RECENT_FORM if missing
        if 'RECENT_FORM' not in df.columns and 'Weighted_Score' in df.columns:
            df['RECENT_FORM'] = df['Weighted_Score']

        return df
    except Exception as e:
        logging.error(f"Failed to load master data: {e}")
        return None

@lru_cache(maxsize=1)
def load_ml_model(model_path: str = MODEL_PATH) -> Optional:#type:ignore
    """
    Load the trained ML model.
    """
    try:
        model_path_with_pkl = f"{model_path}.pkl"
        if os.path.exists(model_path_with_pkl):
            # Set environment variables to handle compatibility issues
            os.environ['SKLEARN_ALLOW_DEPRECATED_SKLEARN_IMPORT_PATH'] = 'true'
            # Add numpy random state fix
            import numpy as np
            # Patch numpy.random.RandomState to handle old pickled models
            original_random_state = np.random.RandomState
            try:
                class CompatRandomState(np.random.RandomState):
                    def __init__(self, *args, **kwargs):
                        if len(args) > 1:
                            # Handle old format by taking only the first argument
                            args = (args[0],)
                        super().__init__(*args, **kwargs)
                np.random.RandomState = CompatRandomState
                
                # Try to load with PyCaret first
                try:
                    return load_model(model_path)
                except Exception as inner_e:
                    logging.warning(f"PyCaret load_model failed: {inner_e}")
                    # Fall back to pickle
                    import pickle
                    with open(model_path_with_pkl, 'rb') as f:
                        return pickle.load(f)
            finally:
                # Restore original RandomState
                np.random.RandomState = original_random_state
        if os.path.exists(model_path):
            # Same approach for model path without .pkl
            os.environ['SKLEARN_ALLOW_DEPRECATED_SKLEARN_IMPORT_PATH'] = 'true'
            try:
                return load_model(model_path)
            except Exception as inner_e:
                logging.warning(f"PyCaret load_model failed: {inner_e}")
                import pickle
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
        logging.error(f"Model not found at: {model_path_with_pkl} or {model_path}")
        logging.info("Please run TRAIN1.py first to create the model")
        return None
    except Exception as e:
        logging.error(f"Failed to load ML model: {e}")
        return None

# Data Preparation
def prepare_data_for_prediction(
    match_players_df: pd.DataFrame,
    master_df: pd.DataFrame
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if master_df is None:
        logging.error("Master dataframe is None")
        return None, None

    # Check if we have the necessary career features
    available_features = [feat for feat in CAREER_FEATURES if feat in master_df.columns]
    if not available_features:
        logging.error("No career features found in master dataframe")
        return None, None

    required_cols = ['Player Name', 'Credits', 'Player Type', 'Team']
    if not all(col in match_players_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in match_players_df.columns]
        logging.error(f"Match players dataframe missing required columns: {missing_cols}")
        return None, None

    df = match_players_df.copy()
    df['Player Name Clean'] = df['Player Name'].str.strip().replace(NAME_CORRECTIONS)
    df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce').fillna(0)

    # Handle IsPlaying column
    if 'IsPlaying' in df.columns:
        playing_mask = df['IsPlaying'].str.upper().fillna('') == 'PLAYING'
        x_factor_mask = df['IsPlaying'].str.upper().fillna('') == 'X_FACTOR_SUBSTITUTE'
        df['Is_X_Factor'] = x_factor_mask
        df = df[playing_mask | x_factor_mask]  # Include both PLAYING and X_Factor players
    else:
        df['Is_X_Factor'] = False

    # Handle lineup order if available
    if 'lineupOrder' in df.columns:
        df['lineupOrder'] = pd.to_numeric(df['lineupOrder'], errors='coerce').fillna(99)
    else:
        df['lineupOrder'] = 99  # Default lineup order if not available

    # Merge with master data
    master_subset = master_df[['Player Name Clean'] + available_features + [col for col in FORM_FEATURES if col in master_df.columns]]
    merged_df = df.merge(master_subset, on='Player Name Clean', how='left')

    # Remove duplicate columns
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    logging.info(f"Columns after merging: {merged_df.columns.tolist()}")

    # Report and remove players with missing stats
    missing_players = merged_df[merged_df[available_features].isnull().any(axis=1)]['Player Name'].tolist()
    if missing_players:
        logging.warning(f"Missing stats for players: {missing_players[:10]}{'...' if len(missing_players) > 10 else ''}")
        logging.info("Skipping these players and continuing with available players")
        merged_df = merged_df[~merged_df[available_features].isnull().any(axis=1)]

    # Prepare features for prediction and information for optimization
    features = merged_df[MATCH_FEATURES + available_features]
    # Remove duplicate columns before prediction
    features = features.loc[:, ~features.columns.duplicated()]
    form_cols = [col for col in FORM_FEATURES if col in merged_df.columns]
    info = merged_df[['Player Name', 'Credits', 'Player Type', 'Team', 'lineupOrder', 'Is_X_Factor'] + form_cols]
    return features, info

# Team Selection
def select_best_team(features: pd.DataFrame, player_info: pd.DataFrame, ml_model, max_credits: float = MAX_CREDITS) -> pd.DataFrame:
    """
    Select the best team using ML predictions and optimization.
    """
    if features.empty or player_info.empty:
        logging.error("Empty features or player info dataframes")
        return pd.DataFrame()

    if ml_model is None:
        logging.error("ML model is None")
        return pd.DataFrame()

    try:
        predictions = predict_model(ml_model, data=features)
        if 'prediction_label' not in predictions.columns:
            logging.error("No prediction_label column found in predictions")
            return pd.DataFrame()

        pred_scores = predictions['prediction_label'].clip(lower=0).rename('Predicted_Score')
        optimization_df = pd.concat([player_info, pred_scores], axis=1)
        optimization_df = optimization_df.dropna(subset=['Credits', 'Predicted_Score', 'Player Name', 'Team', 'Player Type'])
        return select_best_team_optimized_pulp(optimization_df, max_credits)
    except Exception as e:
        logging.error(f"Prediction/Optimization error: {e}")
        return pd.DataFrame()

def select_best_team_optimized_pulp(
    df: pd.DataFrame,
    max_credits: float,
    lineup_bonus: float = 0.10,
    form_weight: float = 0.25,
    recent_form_weight: float = 0.15
) -> pd.DataFrame:
    if df.empty:
        logging.error("Empty dataframe for optimization")
        return pd.DataFrame()

    # Prepare data for optimization
    players = df['Player Name'].tolist()
    data = {
        'credits': df.set_index('Player Name')['Credits'].to_dict(),
        'scores': df.set_index('Player Name')['Predicted_Score'].to_dict(),
        'roles': df.set_index('Player Name')['Player Type'].str.upper().to_dict(),
        'teams': df.set_index('Player Name')['Team'].to_dict(),
        'lineup': df.set_index('Player Name')['lineupOrder'].to_dict(),
        'is_x_factor': df.set_index('Player Name')['Is_X_Factor'].to_dict()
    }

    # Add weighted score and recent form if available
    for form_col in ['Weighted_Score', 'RECENT_FORM']:
        if form_col in df.columns:
            data[form_col.lower()] = df.set_index('Player Name')[form_col].to_dict()
            break
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
        # Form consideration - use weighted_score or recent_form if available
        form_bonus = data.get('weighted_score', data.get('recent_form', {p: 0}))[p] * form_weight
        # Role-specific adjustments
        role_bonus = 0
        if 'ALL' in data['roles'][p] or 'ROUNDER' in data['roles'][p]:
            # All-rounders have more opportunities to score
            role_bonus = 0.20 * base_score
        elif 'BOWL' in data['roles'][p] and lineup_position <= 8:
            # Bowling all-rounders (bowlers who can bat)
            role_bonus = 0.12 * base_score
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
        'BAT': (2, 6, lambda r: 'BAT' in r and 'WICKETKEEPER' not in r),  # More batsmen for balance
        'AR': (1, 6, lambda r: 'ALL' in r or 'ROUNDER' in r),
        'BOWL': (2, 6, lambda r: 'BOWL' in r)  # More bowlers for balance
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

    # X_Factor Substitute constraint: At most one X_Factor player
    x_factor_players = [p for p in players if data['is_x_factor'][p]]
    prob += pulp.lpSum(player_vars[p] for p in x_factor_players) <= 1

    # Ensure X_Factor players are included only if their score > 45
    for p in x_factor_players:
        prob += player_vars[p] * data['scores'][p] >= 50 * player_vars[p]

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=20)  # Set time limit for larger problems
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != 'Optimal':
        logging.warning(f"Optimal solution not found. Using best available solution: {pulp.LpStatus[prob.status]}")

    selected_players = [p for p in players if player_vars[p].varValue > 0.5]
    result_df = df[df['Player Name'].isin(selected_players)].copy()
    result_df['Score_Used'] = result_df['Player Name'].map(lambda p: adjusted_scores[p])
    return result_df

# Captain/Vice-Captain Assignment
def assign_captain_vice_captain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign captain and vice-captain based on adjusted scores.
    """
    if df.empty or 'Predicted_Score' not in df.columns:
        logging.error("Empty dataframe or missing Predicted_Score for captain assignment")
        return df

    df_cvc = df.copy()
    df_cvc['CVC_Score'] = df_cvc['Predicted_Score']

    # Role-specific adjustments
    role_bonuses = {
        'BAT': {'top_order': lambda row: 0.20 if 'BAT' in row['Player Type'].upper() and row.get('lineupOrder', 99) <= 4 else 0},
        'BOWL': {'death_bowler': lambda row: 0.20 if 'BOWL' in row['Player Type'].upper() and row.get('Economy Rate', 0) < 8.0 else 0},
        'ALL': {'dual_threat': lambda row: 0.30 if 'ALL' in row['Player Type'].upper() or 'ROUNDER' in row['Player Type'].upper() else 0},
        'WK': {'batting_wk': lambda row: 0.20 if 'WK' in row['Player Type'].upper() and row.get('lineupOrder', 99) <= 4 else 0}
    }
    for role, bonuses in role_bonuses.items():
        for bonus_func in bonuses.values():
            df_cvc['CVC_Score'] += df_cvc.apply(bonus_func, axis=1) * df_cvc['Predicted_Score']

    # Form-based adjustments
    form_columns = [col for col in ['Weighted_Score', 'RECENT_FORM', 'Dream11_Avg', 'Dream_11_2024_avg'] if col in df_cvc.columns]
    for form_col in form_columns:
        df_cvc['CVC_Score'] += df_cvc[form_col] * 0.10

    # Lineup order considerations
    if 'lineupOrder' in df_cvc.columns:
        df_cvc['CVC_Score'] += df_cvc.apply(
            lambda row: max(0, (5 - min(row.get('lineupOrder', 5), 5)) / 10) * row['Predicted_Score']
            if 'BAT' in row['Player Type'].upper() or 'ALL' in row['Player Type'].upper() else 0,
            axis=1
        )

    captain_vice = df_cvc.nlargest(2, 'CVC_Score')['Player Name'].tolist()
    df['C/VC'] = df['Player Name'].map(
        lambda x: 'C' if x == captain_vice[0] else 'VC' if len(captain_vice) > 1 and x == captain_vice[1] else ''
    )
    return df

# Main Function
def main():
    """
    Main function to execute the application.
    """
    if len(sys.argv) != 2:
        logging.error("Usage: python app1.py <match_number>")
        logging.info("Example: python app1.py 1")
        return

    try:
        match_number = int(sys.argv[1])
        sheet_name = f"Match_{match_number}"
    except ValueError:
        logging.error("Match number must be a valid integer")
        return

    logging.info(f"Loading data for {sheet_name}...")
    master_data = load_master_data()
    if master_data is None:
        logging.error(f"Please ensure {MASTER_FILE_PATH} exists in the current directory")
        return

    ml_model = load_ml_model()
    if ml_model is None:
        logging.error(f"Please run TRAIN1.py first to create the model at {MODEL_PATH}")
        return

    logging.info("Successfully loaded master data and model")

    try:
        if not os.path.exists(INPUT_EXCEL_PATH):
            logging.error(f"Input Excel file not found at: {INPUT_EXCEL_PATH}")
            return

        try:
            match_data = pd.read_excel(INPUT_EXCEL_PATH, sheet_name=sheet_name)
            logging.info("Successfully loaded match data")
        except ValueError:
            logging.error(f"Sheet '{sheet_name}' not found in the Excel file")
            available_sheets = pd.ExcelFile(INPUT_EXCEL_PATH).sheet_names
            logging.info("Available sheets:")
            for sheet in available_sheets:
                logging.info(f"- {sheet}")
            return

        features, info = prepare_data_for_prediction(match_data, master_data)
        if features is not None and info is not None:
            best_team = select_best_team(features, info, ml_model)
            if not best_team.empty:
                final_team = assign_captain_vice_captain(best_team)
                output_filename = f'predicted_team_match_{match_number}.csv'
                # Always save to the user's Downloads folder
                downloads_folder = str(Path.home() / "Downloads")
                output_path = os.path.join(downloads_folder, output_filename)
                output_columns = ['Player Name', 'Team', 'C/VC']
                final_team[output_columns].to_csv(output_path, index=False)

                logging.info("=" * 50)
                logging.info("PREDICTION COMPLETE!")
                logging.info("=" * 50)
                logging.info(f"Team file saved at: {output_path}")
                logging.info("=" * 50)

                # Print summary
                logging.info(f"Credits Used: {final_team['Credits'].sum():.1f} / 100")
                roles_count = final_team['Player Type'].str.upper().apply(
                    lambda x: 'WK' if 'WICKETKEEPER' in x or 'WK' in x else
                              'AR' if 'ALL' in x or 'ROUNDER' in x else
                              'BAT' if 'BAT' in x else 'BOWL'
                ).value_counts()
                logging.info("Role Distribution:")
                for role, count in roles_count.items():
                    logging.info(f"{role}: {count}")

                logging.info("Team Distribution:")
                for team, count in final_team['Team'].value_counts().items():
                    logging.info(f"{team}: {count}")

                captain = final_team[final_team['C/VC'] == 'C']['Player Name'].iloc[0]
                vice_captain = final_team[final_team['C/VC'] == 'VC']['Player Name'].iloc[0]
                logging.info(f"Captain: {captain}")
                logging.info(f"Vice Captain: {vice_captain}")
            else:
                logging.error("No valid team could be selected")
        else:
            logging.error("Failed to prepare data for prediction")
    except Exception as e:
        logging.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
