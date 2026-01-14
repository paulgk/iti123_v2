#!/usr/bin/env python3
"""
Data Organization Script
1. Rename videos to match IDs from match.csv
2. Parse all stroke annotations
3. Filter for Clear (長球) and Smash (殺球) only
4. Generate comprehensive statistics
"""

import os
import pandas as pd
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import re

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
RAW_VIDEOS_DIR = BASE_DIR / "data" / "raw_videos"
MATCH_CSV = ANNOTATIONS_DIR / "match.csv"

def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    if not url or pd.isna(url):
        return None

    # Handle youtube.com/watch?v=VIDEO_ID
    if 'youtube.com' in url:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        if 'v' in query_params:
            return query_params['v'][0]

    # Handle youtu.be/VIDEO_ID
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]

    return None


def rename_videos():
    """Rename videos from YouTube IDs to match IDs"""
    print("=" * 70)
    print("STEP 1: Renaming Videos to Match IDs")
    print("=" * 70)
    print()

    # Load match.csv
    if not MATCH_CSV.exists():
        print(f"❌ ERROR: {MATCH_CSV} not found!")
        return None

    matches_df = pd.read_csv(MATCH_CSV)
    print(f"✅ Loaded {len(matches_df)} matches from match.csv")

    # Create mapping: youtube_id -> match_id
    video_mapping = {}
    for idx, row in matches_df.iterrows():
        youtube_id = extract_youtube_id(row['url'])
        if youtube_id:
            video_mapping[youtube_id] = row['id']

    print(f"✅ Found {len(video_mapping)} YouTube ID mappings")
    print()

    # Get list of video files
    video_files = list(RAW_VIDEOS_DIR.glob('*.mp4'))
    print(f"📁 Found {len(video_files)} video files in {RAW_VIDEOS_DIR}")
    print()

    # Rename videos
    renamed_count = 0
    already_named = 0
    not_found = 0

    for video_file in video_files:
        filename = video_file.stem  # Filename without extension

        # Skip if already renamed to match_id format
        if filename.isdigit():
            already_named += 1
            continue

        # Extract YouTube ID from filename
        youtube_id = filename.split('.')[0]  # Handle files like "VIDEO_ID.f399.mp4.part"

        if youtube_id in video_mapping:
            match_id = video_mapping[youtube_id]
            new_name = f"{match_id:02d}.mp4"
            new_path = RAW_VIDEOS_DIR / new_name

            # Check if target already exists
            if new_path.exists():
                print(f"⚠️  {new_name} already exists, skipping {video_file.name}")
                already_named += 1
            else:
                print(f"✅ Renaming: {video_file.name} -> {new_name}")
                video_file.rename(new_path)
                renamed_count += 1
        else:
            print(f"⚠️  YouTube ID not found in match.csv: {youtube_id}")
            not_found += 1

    print()
    print(f"Summary:")
    print(f"  Renamed: {renamed_count}")
    print(f"  Already named correctly: {already_named}")
    print(f"  Not found in match.csv: {not_found}")
    print()

    return matches_df


def parse_stroke_annotations(matches_df):
    """Parse all stroke annotations and filter for Clear and Smash"""
    print("=" * 70)
    print("STEP 2: Parsing Stroke Annotations")
    print("=" * 70)
    print()

    all_strokes = []

    # Define stroke types we want (excluding wrist smash)
    TARGET_STROKES = {
        '長球': 'Clear',      # Clear
        '殺球': 'Smash',      # Smash
    }

    print(f"Target stroke types:")
    for chinese, english in TARGET_STROKES.items():
        print(f"  {chinese} -> {english}")
    print()

    # Iterate through each match
    for idx, match in matches_df.iterrows():
        match_id = match['id']
        match_folder = match['video']
        match_path = ANNOTATIONS_DIR / match_folder

        if not match_path.exists():
            print(f"⚠️  Match folder not found: {match_folder}")
            continue

        # Process each set
        for set_num in range(1, match['set'] + 1):
            set_file = match_path / f"set{set_num}.csv"

            if not set_file.exists():
                print(f"⚠️  Set file not found: {set_file}")
                continue

            # Read set annotations
            try:
                set_df = pd.read_csv(set_file)

                # Filter for target stroke types
                target_strokes_df = set_df[set_df['type'].isin(TARGET_STROKES.keys())].copy()

                if len(target_strokes_df) > 0:
                    # Add match metadata
                    target_strokes_df['match_id'] = match_id
                    target_strokes_df['match_name'] = match_folder
                    target_strokes_df['set_num'] = set_num
                    target_strokes_df['tournament'] = match['tournament']
                    target_strokes_df['year'] = match['year']
                    target_strokes_df['winner'] = match['winner']
                    target_strokes_df['loser'] = match['loser']

                    # Map Chinese to English
                    target_strokes_df['stroke_type_english'] = target_strokes_df['type'].map(TARGET_STROKES)

                    all_strokes.append(target_strokes_df)

                    print(f"✅ Match {match_id:2d}, Set {set_num}: {len(target_strokes_df):4d} strokes")

            except Exception as e:
                print(f"❌ Error reading {set_file}: {e}")

    if not all_strokes:
        print("❌ No strokes found!")
        return None

    # Combine all strokes
    all_strokes_df = pd.concat(all_strokes, ignore_index=True)

    print()
    print(f"✅ Total strokes extracted: {len(all_strokes_df)}")
    print()

    return all_strokes_df


def generate_statistics(all_strokes_df):
    """Generate comprehensive statistics"""
    print("=" * 70)
    print("STEP 3: Generating Statistics")
    print("=" * 70)
    print()

    # Overall statistics
    print("📊 OVERALL STATISTICS")
    print("-" * 70)
    print(f"Total strokes: {len(all_strokes_df)}")
    print()

    # By stroke type
    print("By Stroke Type:")
    stroke_counts = all_strokes_df['stroke_type_english'].value_counts()
    for stroke_type, count in stroke_counts.items():
        percentage = (count / len(all_strokes_df)) * 100
        print(f"  {stroke_type:10s}: {count:6d} ({percentage:5.2f}%)")
    print()

    # By year
    print("By Year:")
    year_counts = all_strokes_df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:6d} strokes")
    print()

    # By tournament
    print("By Tournament (Top 10):")
    tournament_counts = all_strokes_df['tournament'].value_counts().head(10)
    for tournament, count in tournament_counts.items():
        print(f"  {tournament:50s}: {count:6d} strokes")
    print()

    # By match
    print("By Match (Top 10 matches with most strokes):")
    match_counts = all_strokes_df.groupby(['match_id', 'match_name']).size().sort_values(ascending=False).head(10)
    for (match_id, match_name), count in match_counts.items():
        print(f"  Match {match_id:2d} ({match_name[:50]:50s}): {count:4d} strokes")
    print()

    # By player
    print("By Player (Top 10):")
    player_counts = all_strokes_df['player'].value_counts().head(10)
    for player, count in player_counts.items():
        print(f"  {player:30s}: {count:6d} strokes")
    print()

    # Stroke type by player (top 5 players)
    print("Stroke Type Distribution by Player (Top 5 players):")
    top_players = player_counts.head(5).index
    for player in top_players:
        player_strokes = all_strokes_df[all_strokes_df['player'] == player]
        clear_count = len(player_strokes[player_strokes['stroke_type_english'] == 'Clear'])
        smash_count = len(player_strokes[player_strokes['stroke_type_english'] == 'Smash'])
        total = len(player_strokes)
        print(f"  {player:30s}: Clear={clear_count:4d} ({clear_count/total*100:5.1f}%), Smash={smash_count:4d} ({smash_count/total*100:5.1f}%)")
    print()

    # Hit area statistics
    print("Hit Area Distribution:")
    hit_area_counts = all_strokes_df['hit_area'].value_counts().sort_index()
    for area, count in hit_area_counts.items():
        percentage = (count / len(all_strokes_df)) * 100
        print(f"  Area {area}: {count:6d} ({percentage:5.2f}%)")
    print()

    # Landing area statistics
    print("Landing Area Distribution:")
    landing_area_counts = all_strokes_df['landing_area'].value_counts().sort_index()
    for area, count in landing_area_counts.items():
        percentage = (count / len(all_strokes_df)) * 100
        print(f"  Area {area}: {count:6d} ({percentage:5.2f}%)")
    print()

    # Height statistics
    print("Hit Height Distribution:")
    hit_height_counts = all_strokes_df['hit_height'].value_counts().sort_index()
    height_labels = {1.0: 'High', 2.0: 'Medium', 3.0: 'Low'}
    for height, count in hit_height_counts.items():
        percentage = (count / len(all_strokes_df)) * 100
        label = height_labels.get(height, 'Unknown')
        print(f"  {label} (height={height}): {count:6d} ({percentage:5.2f}%)")
    print()

    # Save detailed statistics to CSV
    output_dir = BASE_DIR / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_file = output_dir / "stroke_statistics.csv"
    all_strokes_df.to_csv(stats_file, index=False)
    print(f"✅ Detailed stroke data saved to: {stats_file}")

    # Save summary statistics
    summary_file = output_dir / "summary_statistics.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SHUTTLESET STROKE STATISTICS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total strokes (Clear + Smash only): {len(all_strokes_df)}\n\n")

        f.write("By Stroke Type:\n")
        for stroke_type, count in stroke_counts.items():
            percentage = (count / len(all_strokes_df)) * 100
            f.write(f"  {stroke_type:10s}: {count:6d} ({percentage:5.2f}%)\n")

        f.write("\n")
        f.write(f"Number of matches: {all_strokes_df['match_id'].nunique()}\n")
        f.write(f"Number of unique players: {all_strokes_df['player'].nunique()}\n")
        f.write(f"Years covered: {all_strokes_df['year'].min()} - {all_strokes_df['year'].max()}\n")

    print(f"✅ Summary statistics saved to: {summary_file}")
    print()

    return all_strokes_df


def main():
    """Main function"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: Data Organization & Statistics".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    # Step 1: Rename videos
    matches_df = rename_videos()

    if matches_df is None:
        print("❌ Failed to load match data. Exiting.")
        return

    # Step 2: Parse annotations
    all_strokes_df = parse_stroke_annotations(matches_df)

    if all_strokes_df is None:
        print("❌ Failed to parse annotations. Exiting.")
        return

    # Step 3: Generate statistics
    generate_statistics(all_strokes_df)

    print("=" * 70)
    print("✅ DATA ORGANIZATION COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review the statistics in: outputs/reports/")
    print("2. Proceed with video clip extraction")
    print("3. Run: python src/data_processing/extract_clips.py")
    print()


if __name__ == "__main__":
    main()
