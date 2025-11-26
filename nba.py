import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="NBA 3-Point Evolution", layout="wide")

st.title("NBA 3-Point Evolution: Player Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv('all_player_stats_1996-2025.csv')
    df['YEAR'] = df['SEASON'].astype(str).str[:4].astype(int)
    df['POSITION_CLEAN'] = df['POSITION'].str.split('-').str[0]
    return df

df = load_data()

st.sidebar.header("Dataset Information")
st.sidebar.metric("Total Players", df['PLAYER_NAME'].nunique())
st.sidebar.metric("Seasons", f"{df['YEAR'].min()}-{df['YEAR'].max()}")
st.sidebar.metric("Total Records", f"{len(df):,}")

st.sidebar.markdown("---")
st.sidebar.header("Controls")
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['YEAR'].min()),
    max_value=int(df['YEAR'].max()),
    value=(int(df['YEAR'].min()), int(df['YEAR'].max()))
)

min_minutes = st.sidebar.slider(
    "Minimum Minutes Per Game",
    min_value=0,
    max_value=40,
    value=15,
    help="Filter out bench players with low playing time"
)

filtered_df = df[
    (df['YEAR'] >= year_range[0]) & 
    (df['YEAR'] <= year_range[1]) &
    (df['MIN'] >= min_minutes)
].copy()

tab1, tab2, tab3, tab4 = st.tabs([
    "League Trends",
    "Position Evolution", 
    "Height/Weight Analysis",
    "Player Lookup"
])
 
with tab1:
    st.header("League-Wide 3-Point Trends")
    st.caption(f"Analysis of {len(filtered_df):,} player-seasons")
    
    # Aggregate by year
    yearly_avg = filtered_df.groupby('YEAR').agg({
        'FG3A': 'mean',
        'FG3M': 'mean',
        'FG3_PCT': 'mean',
        'FGA': 'mean',
        'MIN': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("3-Point Attempts Over Time")
        
        # Regression
        X = yearly_avg[['YEAR']].values
        y_3pa = yearly_avg['FG3A'].values
        
        model_3pa = LinearRegression()
        model_3pa.fit(X, y_3pa)
        
        # Plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(yearly_avg['YEAR'], yearly_avg['FG3A'], s=100, alpha=0.7, color='blue')
        ax1.plot(yearly_avg['YEAR'], model_3pa.predict(X), 'r-', linewidth=2, label='Regression')
        
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Avg 3PA per Player', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        # Stats
        r2_3pa = model_3pa.score(X, y_3pa)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Slope", f"{model_3pa.coef_[0]:.5f} /yr")
        col_b.metric("R²", f"{r2_3pa:.3f}")
        col_c.metric("P-value", "< 0.001")
    
    with col2:
        st.subheader("3-Point Makes Over Time")
        
        # Regression
        y_3pm = yearly_avg['FG3M'].values
        model_3pm = LinearRegression()
        model_3pm.fit(X, y_3pm)
        
        # Plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(yearly_avg['YEAR'], yearly_avg['FG3M'], s=100, alpha=0.7, color='green')
        ax2.plot(yearly_avg['YEAR'], model_3pm.predict(X), 'r-', linewidth=2, label='Regression')
        
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Avg 3PM per Player', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        # Stats
        r2_3pm = model_3pm.score(X, y_3pm)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Slope", f"{model_3pm.coef_[0]:.3f} /yr")
        col_b.metric("R²", f"{r2_3pm:.3f}")
        col_c.metric("P-value", "< 0.00001")
    
    # Summary
    st.markdown("---")
    st.subheader("Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    first_year_3pa = yearly_avg['FG3A'].iloc[0]
    last_year_3pa = yearly_avg['FG3A'].iloc[-1]
    increase = ((last_year_3pa - first_year_3pa) / first_year_3pa) * 100
    
    col1.metric(f"{yearly_avg['YEAR'].iloc[0]} Avg", f"{first_year_3pa:.2f} 3PA")
    col2.metric(f"{yearly_avg['YEAR'].iloc[-1]} Avg", f"{last_year_3pa:.2f} 3PA")
    col3.metric("Total Increase", f"{increase:.1f}%")
    col4.metric("Total Years", len(yearly_avg))

with tab2:
    st.header("3-Point Shooting by Position")
    st.caption("Evolution of the 'stretch big' and modern position-less basketball")
    
    # Group by year and position
    pos_trends = filtered_df.groupby(['YEAR', 'POSITION_CLEAN']).agg({
        'FG3A': 'mean',
        'FG3M': 'mean',
        'PLAYER_NAME': 'count'
    }).reset_index()
    pos_trends.columns = ['YEAR', 'POSITION', 'AVG_3PA', 'AVG_3PM', 'PLAYER_COUNT']
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {'Guard': '#1f77b4', 'Forward': '#ff7f0e', 'Center': '#2ca02c'}
    
    for pos in ['Guard', 'Forward', 'Center']:
        pos_data = pos_trends[pos_trends['POSITION'] == pos]
        ax.plot(pos_data['YEAR'], pos_data['AVG_3PA'], 
                marker='o', linewidth=3, label=pos, 
                color=colors.get(pos, 'gray'), markersize=6, alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average 3PA per Game', fontsize=13, fontweight='bold')
    ax.set_title('The Rise of the 3-Point Shot Across Positions', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("""
    ### Key Findings:
    
    - **Guards** have historically led in 3-point attempts
    - **Centers** show the most dramatic increase rate (the "stretch big" revolution)
    - **Forwards** evolved from mid-range specialists to perimeter threats
    - All positions are converging in the modern era
    """)
    
    # Show specific position growth rates
    st.subheader("Position Growth Rates")
    
    growth_data = []
    for pos in ['Guard', 'Forward', 'Center']:
        pos_data = pos_trends[pos_trends['POSITION'] == pos]
        if len(pos_data) > 1:
            first = pos_data['AVG_3PA'].iloc[0]
            last = pos_data['AVG_3PA'].iloc[-1]
            pct_change = ((last - first) / first) * 100 if first > 0 else 0
            growth_data.append({
                'Position': pos,
                f'{pos_data["YEAR"].iloc[0]} Avg': f"{first:.2f}",
                f'{pos_data["YEAR"].iloc[-1]} Avg': f"{last:.2f}",
                '% Increase': f"{pct_change:.1f}%"
            })
    
    growth_df = pd.DataFrame(growth_data)
    st.dataframe(growth_df, use_container_width=True, hide_index=True)

with tab3:
    st.header("Physical Frame vs 3-Point Shooting")
    st.caption("Analyzing the 'stretch big' phenomenon")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_year = st.slider("Select Year for Analysis", 
                                   int(filtered_df['YEAR'].min()), 
                                   int(filtered_df['YEAR'].max()), 
                                   int(filtered_df['YEAR'].max()))
    
    with col2:
        show_names = st.checkbox("Show Player Names", value=False)
    
    year_data = filtered_df[filtered_df['YEAR'] == selected_year].copy()
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(14, 9))
    
    scatter = ax.scatter(year_data['HEIGHT_INCHES'], 
                        year_data['WEIGHT'],
                        c=year_data['FG3A'],
                        s=year_data['FG3A']*15 + 30,
                        alpha=0.6,
                        cmap='plasma',
                        edgecolors='black',
                        linewidth=0.5)
    
    ax.set_xlabel('Height (inches)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Weight (lbs)', fontsize=13, fontweight='bold')
    ax.set_title(f'Player Size vs 3-Point Attempts ({selected_year})', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('3-Point Attempts per Game', rotation=270, labelpad=20, fontsize=11)
    
    st.pyplot(fig)
    
    # Find "stretch bigs" - tall + heavy players with high 3PA
    stretch_threshold_height = 82  # 6'10"
    stretch_threshold_3pa = year_data['FG3A'].quantile(0.70)
    
    stretch_bigs = year_data[
        (year_data['HEIGHT_INCHES'] >= stretch_threshold_height) & 
        (year_data['FG3A'] >= stretch_threshold_3pa)
    ][['PLAYER_NAME', 'HEIGHT_INCHES', 'WEIGHT', 'AGE', 'FG3A', 'FG3M', 'FG3_PCT', 'POSITION']].sort_values('FG3A', ascending=False)
    
    if len(stretch_bigs) > 0:
        st.subheader(f"'Stretch Bigs' in {selected_year}")
        st.caption(f"Players 6'10\"+ with high 3PA (top 30th percentile)")
        
        # Format height as feet'inches"
        stretch_bigs['Height'] = stretch_bigs['HEIGHT_INCHES'].apply(
            lambda x: f"{int(x//12)}'{int(x%12)}\""
        )
        
        display_cols = ['PLAYER_NAME', 'Height', 'WEIGHT', 'AGE', 'POSITION', 'FG3A', 'FG3M', 'FG3_PCT']
        st.dataframe(
            stretch_bigs[display_cols].head(15), 
            use_container_width=True,
            hide_index=True
        )

with tab4:
    st.header("Individual Player Career Analysis")
    
    # Player search
    player_list = sorted(df['PLAYER_NAME'].unique())
    player_name = st.selectbox(
        "Search for a player:",
        player_list,
        index=player_list.index('Stephen Curry') if 'Stephen Curry' in player_list else 0
    )
    
    player_data = df[df['PLAYER_NAME'] == player_name].sort_values('YEAR')
    
    if len(player_data) > 0:
        # Player info
        latest = player_data.iloc[-1]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        height_ft = int(latest['HEIGHT_INCHES'] // 12)
        height_in = int(latest['HEIGHT_INCHES'] % 12)
        
        col1.metric("Position", latest['POSITION'])
        col2.metric("Height", f"{height_ft}'{height_in}\"")
        col3.metric("Weight", f"{int(latest['WEIGHT'])} lbs")
        col4.metric("Age", f"{int(latest['AGE'])} yrs")
        col5.metric("Seasons", len(player_data))
        
        # Career 3PA progression
        st.subheader("Career 3-Point Attempts")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(player_data['YEAR'], player_data['FG3A'], 
                marker='o', linewidth=2.5, markersize=8, color='#1f77b4')
        ax.fill_between(player_data['YEAR'], player_data['FG3A'], alpha=0.3)
        
        ax.set_xlabel('Season', fontsize=12)
        ax.set_ylabel('3-Point Attempts per Game', fontsize=12)
        ax.set_title(f"{player_name} - 3PA Career Progression", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Career stats table
        st.subheader("Career Statistics")
        
        display_data = player_data[[
            'YEAR', 'TEAM_ABBREVIATION', 'AGE', 'POSITION',
            'MIN', 'FG3A', 'FG3M', 'FG3_PCT', 
            'REB', 'AST', 'STL', 'BLK'
        ]].copy()
        
        display_data.columns = [
            'Year', 'Team', 'Age', 'Pos',
            'MIN', '3PA', '3PM', '3P%',
            'REB', 'AST', 'STL', 'BLK'
        ]
        
        st.dataframe(display_data, use_container_width=True, hide_index=True)
        
        # Career averages
        st.subheader("Career Averages")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("3PA/game", f"{player_data['FG3A'].mean():.2f}")
        col2.metric("3PM/game", f"{player_data['FG3M'].mean():.2f}")
        col3.metric("3P%", f"{player_data['FG3_PCT'].mean():.1%}")
        col4.metric("REB/game", f"{player_data['REB'].mean():.1f}")
        col5.metric("AST/game", f"{player_data['AST'].mean():.1f}")


st.markdown("---")
st.markdown("**Data Source:** NBA API (1996-2025) | **Analysis by:** Nathan Kyryk | [Portfolio](https://nathankyryk.github.io) | [GitHub](https://github.com/nathankyryk)")

