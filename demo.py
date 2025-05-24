import json
import pandas as pd
from collections import defaultdict

# Load match JSON file
with open("ipl_json\\\\1473499.json", 'r') as f:
    match_data = json.load(f)

balls_per_over = match_data['info'].get('balls_per_over', 6)

# Initialize stats dictionaries
batting_stats = defaultdict(lambda: {
    'runs_scored': 0, 'balls_faced': 0, '4s': 0, '6s': 0,
    '1s': 0, '2s': 0, '3s': 0, 'dot_balls': 0,
    'not_out': 1, 'out_kind': None, 'out_to': None
})
bowling_stats = defaultdict(lambda: {
    'legal_balls': 0, 'runs_conceded': 0, 'wickets': 0,
    'dot_balls': 0, 'wides': 0, 'no_balls': 0, 'extras_conceded': 0, 'total_runs_conceded': 0
})

# Parse each delivery in each innings
for inn in match_data.get('innings', []):
    for over in inn['overs']:
        for delivery in over['deliveries']:
            batter = delivery['batter']
            bowler = delivery['bowler']
            extras = delivery.get('extras', {})

            # Ensure player entries exist
            _ = batting_stats[batter]
            _ = bowling_stats[bowler]

            # --- Batting stats updates ---
            # Check if ball was legal for batting (exclude wides and no-balls)
            if 'wides' not in extras:
                batting_stats[batter]['balls_faced'] += 1
                runs = delivery['runs']['batter']
                batting_stats[batter]['runs_scored'] += runs
                # Update boundary counts and singles/doubles/triples
                if runs == 4:
                    batting_stats[batter]['4s'] += 1
                elif runs == 6:
                    batting_stats[batter]['6s'] += 1
                elif runs == 3:
                    batting_stats[batter]['3s'] += 1
                elif runs == 2:
                    batting_stats[batter]['2s'] += 1
                elif runs == 1:
                    batting_stats[batter]['1s'] += 1
                # Count dot ball for batting
                if runs == 0:
                    batting_stats[batter]['dot_balls'] += 1

            # Handle wickets/dismissals
            if 'wickets' in delivery:
                for w in delivery['wickets']:
                    player_out = w['player_out']
                    kind = w['kind']
                    dismiss_bowler = bowler  # bowler of the delivery
                    # Record dismissal in batting stats
                    _ = batting_stats[player_out]
                    batting_stats[player_out]['out_kind'] = kind
                    batting_stats[player_out]['out_to'] = dismiss_bowler
                    batting_stats[player_out]['not_out'] = 0
                    # Credit wicket to bowler if kind is none of {'retr hurt', 'run out'}
                    if kind != 'run out' and kind != 'retired hurt':
                        bowling_stats[bowler]['wickets'] += 1

            # --- Bowling stats updates ---
            # Count wides and no-balls
            if 'wides' in extras:
                bowling_stats[bowler]['wides'] += delivery['extras']['wides']
                # Add all extra runs from the wide
                bowling_stats[bowler]['extras_conceded'] += delivery['runs']['extras']
                bowling_stats[bowler]['total_runs_conceded'] += delivery['extras']['wides']
            if 'noballs' in extras:
                bowling_stats[bowler]['no_balls'] += 1
                bowling_stats[bowler]['extras_conceded'] += delivery['runs']['extras']
                bowling_stats[bowler]['runs_conceded'] += delivery['runs']['batter']
                bowling_stats[bowler]['total_runs_conceded'] += delivery['extras']['noballs'] + delivery['runs']['batter']

            # Process legal delivery for bowler
            if 'wides' not in extras and 'noballs' not in extras:
                bowling_stats[bowler]['legal_balls'] += 1
                runs_batter = delivery['runs']['batter']
                total_runs = delivery['runs']['total']
                # Exclude byes/legbyes from runs conceded by bowler
                legbyes = extras.get('legbyes', 0)
                byes = extras.get('byes', 0)
                runs_conceded = total_runs - legbyes - byes
                bowling_stats[bowler]['runs_conceded'] += runs_conceded
                bowling_stats[bowler]['total_runs_conceded'] += runs_conceded
                # Count dot ball for bowler (no runs by batter, no wicket)
                if runs_batter == 0 and 'wickets' not in delivery:
                    bowling_stats[bowler]['dot_balls'] += 1
            

# Convert batting stats to DataFrame
bat_df = pd.DataFrame.from_dict(batting_stats, orient='index').reset_index()
bat_df.rename(columns={'index': 'player'}, inplace=True)

# Compute derived batting metrics
# Strike rate
bat_df['strike_rate'] = (bat_df['runs_scored'] / bat_df['balls_faced'] * 100).round(2)

# Fill NaN or infinite values for players with zero balls faced
bat_df['strike_rate'].fillna(0, inplace=True)

# Convert not_out to integer 0/1
bat_df['not_out'] = bat_df['not_out'].astype(int)
# Replace None in out_kind/out_to with empty string for CSV clarity
bat_df['out_kind'] = bat_df['out_kind'].fillna('')
bat_df['out_to'] = bat_df['out_to'].fillna('')

# Convert bowling stats to DataFrame
bowl_df = pd.DataFrame.from_dict(bowling_stats, orient='index').reset_index()
bowl_df.rename(columns={'index': 'player'}, inplace=True)

# Compute overs bowled, economy
bowl_df['overs'] = bowl_df['legal_balls'] // balls_per_over \
                   + (bowl_df['legal_balls'] % balls_per_over) / 10.0
# Economy: runs per over
bowl_df['economy_rate'] = (bowl_df['total_runs_conceded'] * balls_per_over / bowl_df['legal_balls']).round(2)
# Bowling average (runs per wicket)


# Save DataFrames to CSV
bat_df.to_csv("batting_stats.csv", index=False)
bowl_df.to_csv("bowling_stats.csv", index=False)
