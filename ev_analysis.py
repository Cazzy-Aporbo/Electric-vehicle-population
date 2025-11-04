import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#172226', '#000000', '#5F5CA4', '#476F75', '#597C82', '#B2E4D9']
GRADIENT = ['#172226', '#2A3A3E', '#3D5156', '#476F75', '#597C82', '#6B8A8F', '#8BA8A8', '#A9C3C1', '#B2E4D9']
OUTPUT_PATH = '/Electric_Vehicles/'

plt.style.use('dark_background')
sns.set_palette(COLORS)

def quantum_clean(df):
    df = df.copy()
    
    df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')
    df['Postal Code'] = pd.to_numeric(df['Postal Code'], errors='coerce')
    
    df = df[(df['Model Year'] >= 1999) & (df['Model Year'] <= 2026)]
    df = df[df['State'].notna()]
    
    df['EV_Binary'] = (df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)').astype(int)
    df['CAFV_Binary'] = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].str.contains('Eligible', na=False).astype(int)
    
    df['Make_Model'] = df['Make'] + ' ' + df['Model']
    df['Year_Category'] = pd.cut(df['Model Year'], bins=[1998, 2014, 2018, 2022, 2027], 
                                   labels=['Pioneer', 'Growth', 'Acceleration', 'Hyperdrive'])
    
    return df

def temporal_cascade(df):
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor('#0A0F10')
    
    yearly = df.groupby(['Model Year', 'Electric Vehicle Type']).size().unstack(fill_value=0)
    
    ax = axes[0, 0]
    yearly.plot(kind='area', stacked=True, ax=ax, color=[COLORS[2], COLORS[4]], alpha=0.8)
    ax.set_facecolor('#0F1415')
    ax.set_title('Temporal Adoption Cascade', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Registrations', fontsize=12, color=COLORS[5])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4])
    
    ax = axes[0, 1]
    top_makes = df['Make'].value_counts().head(10)
    wedges, texts, autotexts = ax.pie(top_makes.values, labels=top_makes.index, 
                                        autopct='%1.1f%%', startangle=90, 
                                        colors=GRADIENT, wedgeprops={'linewidth': 2, 'edgecolor': '#0A0F10'})
    ax.set_title('Market Dominance Matrix', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    for text in texts + autotexts:
        text.set_color(COLORS[5])
        text.set_fontsize(10)
    
    ax = axes[1, 0]
    geo_dist = df.groupby('County').size().sort_values(ascending=False).head(15)
    bars = ax.barh(range(len(geo_dist)), geo_dist.values, 
                   color=[GRADIENT[i % len(GRADIENT)] for i in range(len(geo_dist))])
    ax.set_yticks(range(len(geo_dist)))
    ax.set_yticklabels(geo_dist.index, fontsize=10, color=COLORS[5])
    ax.set_xlabel('Registrations', fontsize=12, color=COLORS[5])
    ax.set_title('Geographic Density Hotspots', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_facecolor('#0F1415')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
    
    for i, (bar, val) in enumerate(zip(bars, geo_dist.values)):
        ax.text(val, i, f' {val:,}', va='center', ha='left', color=COLORS[5], fontsize=9)
    
    ax = axes[1, 1]
    cafv_trend = df.groupby(['Model Year', 'CAFV_Binary']).size().unstack(fill_value=0)
    cafv_pct = cafv_trend.div(cafv_trend.sum(axis=1), axis=0) * 100
    
    ax.plot(cafv_pct.index, cafv_pct[1], linewidth=3, color=COLORS[5], marker='o', 
            markersize=6, markerfacecolor=COLORS[4], markeredgecolor=COLORS[5], markeredgewidth=2)
    ax.fill_between(cafv_pct.index, cafv_pct[1], alpha=0.3, color=COLORS[4])
    ax.set_facecolor('#0F1415')
    ax.set_title('CAFV Eligibility Evolution', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Eligibility %', fontsize=12, color=COLORS[5])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}temporal_cascade.png', dpi=300, facecolor='#0A0F10', edgecolor='none', bbox_inches='tight')
    plt.close()

def dimensional_reduction(df):
    model_features = df.groupby('Make_Model').agg({
        'Model Year': 'mean',
        'EV_Binary': 'mean',
        'CAFV_Binary': 'mean',
        'VIN (1-10)': 'count'
    }).rename(columns={'VIN (1-10)': 'Count'})
    
    model_features = model_features[model_features['Count'] >= 50].copy()
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_features)
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.patch.set_facecolor('#0A0F10')
    
    ax = axes[0]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        s=model_features['Count']*2, 
                        c=model_features['CAFV_Binary'],
                        cmap='cool', alpha=0.7, edgecolors=COLORS[4], linewidth=1.5)
    
    top_models = model_features.nlargest(15, 'Count')
    for idx in top_models.index:
        if idx in model_features.index:
            pos = model_features.index.get_loc(idx)
            ax.annotate(idx, (coords[pos, 0], coords[pos, 1]), 
                       fontsize=8, color=COLORS[5], alpha=0.9,
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_facecolor('#0F1415')
    ax.set_title('Dimensional Collapse: Model Space', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12, color=COLORS[5])
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12, color=COLORS[5])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('CAFV Eligibility', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[1]
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    
    for i in range(5):
        mask = clusters == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  s=model_features.loc[mask, 'Count']*2,
                  c=[GRADIENT[i*2]], alpha=0.7, edgecolors=COLORS[4], 
                  linewidth=1.5, label=f'Cluster {i+1}')
    
    ax.scatter(pca.transform(scaler.transform(kmeans.cluster_centers_))[:, 0],
              pca.transform(scaler.transform(kmeans.cluster_centers_))[:, 1],
              s=500, c=COLORS[1], marker='X', edgecolors=COLORS[5], linewidth=3,
              label='Centroids', zorder=10)
    
    ax.set_facecolor('#0F1415')
    ax.set_title('Quantum Clustering: Fleet Taxonomy', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12, color=COLORS[5])
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12, color=COLORS[5])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4], loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}dimensional_reduction.png', dpi=300, facecolor='#0A0F10', edgecolor='none', bbox_inches='tight')
    plt.close()

def statistical_manifold(df):
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor('#0A0F10')
    
    ax = axes[0, 0]
    year_stats = df.groupby('Model Year').agg({
        'VIN (1-10)': 'count',
        'EV_Binary': 'mean',
        'CAFV_Binary': 'mean'
    })
    
    ax2 = ax.twinx()
    
    bars = ax.bar(year_stats.index, year_stats['VIN (1-10)'], 
                  color=COLORS[4], alpha=0.6, edgecolor=COLORS[5], linewidth=1.5)
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Registrations', fontsize=12, color=COLORS[5])
    ax.tick_params(axis='y', labelcolor=COLORS[5])
    
    line1 = ax2.plot(year_stats.index, year_stats['EV_Binary']*100, 
                     color=COLORS[5], linewidth=3, marker='o', markersize=6, 
                     label='BEV %', markeredgecolor=COLORS[4], markeredgewidth=2)
    line2 = ax2.plot(year_stats.index, year_stats['CAFV_Binary']*100, 
                     color=COLORS[2], linewidth=3, marker='s', markersize=6, 
                     label='CAFV %', markeredgecolor=COLORS[4], markeredgewidth=2)
    ax2.set_ylabel('Percentage', fontsize=12, color=COLORS[5])
    ax2.tick_params(axis='y', labelcolor=COLORS[5])
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4])
    
    ax.set_facecolor('#0F1415')
    ax2.set_facecolor('#0F1415')
    ax.set_title('Dual-Axis Convergence Analysis', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.grid(True, alpha=0.15, color=COLORS[4])
    
    ax = axes[0, 1]
    
    make_year = df.groupby(['Model Year', 'Make']).size().unstack(fill_value=0)
    top_makes = df['Make'].value_counts().head(8).index
    make_year_top = make_year[top_makes]
    
    bottom = np.zeros(len(make_year_top))
    for i, make in enumerate(top_makes):
        ax.bar(make_year_top.index, make_year_top[make], bottom=bottom,
               color=GRADIENT[i], alpha=0.85, edgecolor=COLORS[0], linewidth=0.5,
               label=make)
        bottom += make_year_top[make].values
    
    ax.set_facecolor('#0F1415')
    ax.set_title('Market Share Stratification', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Registrations', fontsize=12, color=COLORS[5])
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4], ncol=2, fontsize=9)
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    
    ax = axes[1, 0]
    
    type_year = df.groupby(['Year_Category', 'Electric Vehicle Type']).size().unstack()
    type_year_pct = type_year.div(type_year.sum(axis=1), axis=0) * 100
    
    x = np.arange(len(type_year_pct.index))
    width = 0.35
    
    ax.bar(x - width/2, type_year_pct.iloc[:, 0], width, 
           color=COLORS[2], alpha=0.8, edgecolor=COLORS[4], linewidth=2,
           label=type_year_pct.columns[0])
    ax.bar(x + width/2, type_year_pct.iloc[:, 1], width,
           color=COLORS[4], alpha=0.8, edgecolor=COLORS[5], linewidth=2,
           label=type_year_pct.columns[1])
    
    ax.set_xticks(x)
    ax.set_xticklabels(type_year_pct.index, fontsize=11, color=COLORS[5])
    ax.set_ylabel('Percentage', fontsize=12, color=COLORS[5])
    ax.set_title('Era-Based Technology Distribution', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_facecolor('#0F1415')
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    
    ax = axes[1, 1]
    
    model_counts = df['Make_Model'].value_counts().head(20)
    
    angles = np.linspace(0, 2*np.pi, len(model_counts), endpoint=False).tolist()
    values = model_counts.values.tolist()
    angles += angles[:1]
    values += values[:1]
    
    ax = plt.subplot(2, 2, 4, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2, color=COLORS[5], markersize=6, 
            markerfacecolor=COLORS[4], markeredgecolor=COLORS[5], markeredgewidth=2)
    ax.fill(angles, values, alpha=0.25, color=COLORS[4])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(model_counts.index, fontsize=7, color=COLORS[5])
    ax.set_facecolor('#0F1415')
    ax.set_title('Radial Model Popularity', fontsize=16, color=COLORS[5], pad=30, weight='bold')
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}statistical_manifold.png', dpi=300, facecolor='#0A0F10', edgecolor='none', bbox_inches='tight')
    plt.close()

def advanced_distribution(df):
    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor('#0A0F10')
    
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    pivot = df.pivot_table(values='VIN (1-10)', 
                          index='County', 
                          columns='Model Year', 
                          aggfunc='count', 
                          fill_value=0)
    
    top_counties = df['County'].value_counts().head(20).index
    pivot_top = pivot.loc[top_counties].T
    
    sns.heatmap(pivot_top, cmap='YlGnBu', ax=ax1, cbar_kws={'label': 'Registrations'},
                linewidths=0.5, linecolor=COLORS[0], alpha=0.9)
    ax1.set_facecolor('#0F1415')
    ax1.set_title('Spatiotemporal Registration Heatmap', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax1.set_xlabel('County', fontsize=12, color=COLORS[5])
    ax1.set_ylabel('Model Year', fontsize=12, color=COLORS[5])
    ax1.tick_params(colors=COLORS[5])
    
    cbar = ax1.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    cbar.set_label('Registrations', color=COLORS[5])
    
    ax2 = fig.add_subplot(gs[0, 2])
    
    recent = df[df['Model Year'] >= 2020]
    type_dist = recent['Electric Vehicle Type'].value_counts()
    
    colors_pie = [COLORS[2], COLORS[4]]
    wedges, texts, autotexts = ax2.pie(type_dist.values, 
                                         labels=type_dist.index,
                                         autopct='%1.1f%%',
                                         colors=colors_pie,
                                         startangle=90,
                                         wedgeprops={'linewidth': 2, 'edgecolor': '#0A0F10'},
                                         textprops={'color': COLORS[5]})
    ax2.set_title('Recent Fleet\nComposition\n(2020+)', fontsize=12, color=COLORS[5], weight='bold')
    
    ax3 = fig.add_subplot(gs[1, 2])
    
    cafv_dist = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts().head(5)
    bars = ax3.barh(range(len(cafv_dist)), cafv_dist.values,
                    color=[GRADIENT[i*2] for i in range(len(cafv_dist))],
                    edgecolor=COLORS[4], linewidth=1.5)
    ax3.set_yticks(range(len(cafv_dist)))
    ax3.set_yticklabels([label[:30] + '...' if len(label) > 30 else label 
                         for label in cafv_dist.index], fontsize=8, color=COLORS[5])
    ax3.set_xlabel('Count', fontsize=10, color=COLORS[5])
    ax3.set_title('CAFV Status\nDistribution', fontsize=12, color=COLORS[5], weight='bold')
    ax3.set_facecolor('#0F1415')
    ax3.grid(True, alpha=0.15, color=COLORS[4], axis='x')
    ax3.tick_params(colors=COLORS[5])
    
    ax4 = fig.add_subplot(gs[2, :])
    
    city_counts = df['City'].value_counts().head(25)
    
    bars = ax4.bar(range(len(city_counts)), city_counts.values,
                   color=[GRADIENT[i % len(GRADIENT)] for i in range(len(city_counts))],
                   edgecolor=COLORS[4], linewidth=1.5, alpha=0.85)
    
    ax4.set_xticks(range(len(city_counts)))
    ax4.set_xticklabels(city_counts.index, rotation=45, ha='right', fontsize=10, color=COLORS[5])
    ax4.set_ylabel('Registrations', fontsize=12, color=COLORS[5])
    ax4.set_title('Urban EV Concentration: Top 25 Cities', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax4.set_facecolor('#0F1415')
    ax4.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax4.tick_params(colors=COLORS[5])
    
    for i, (bar, val) in enumerate(zip(bars, city_counts.values)):
        ax4.text(i, val, f'{val:,}', ha='center', va='bottom', 
                color=COLORS[5], fontsize=8, rotation=0)
    
    plt.savefig(f'{OUTPUT_PATH}advanced_distribution.png', dpi=300, facecolor='#0A0F10', edgecolor='none', bbox_inches='tight')
    plt.close()

def correlation_nexus(df):
    numeric_df = df[['Model Year', 'EV_Binary', 'CAFV_Binary']].copy()
    
    make_encoded = pd.get_dummies(df['Make'], prefix='Make').astype(int)
    top_makes = df['Make'].value_counts().head(10).index
    make_features = make_encoded[[f'Make_{make}' for make in top_makes if f'Make_{make}' in make_encoded.columns]]
    
    analysis_df = pd.concat([numeric_df, make_features], axis=1)
    
    corr_matrix = analysis_df.corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.patch.set_facecolor('#0A0F10')
    
    ax = axes[0]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, ax=ax, 
                linewidths=1, linecolor=COLORS[0],
                cbar_kws={'label': 'Correlation Coefficient'},
                annot_kws={'size': 8})
    ax.set_facecolor('#0F1415')
    ax.set_title('Feature Correlation Nexus', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.tick_params(colors=COLORS[5])
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    cbar.set_label('Correlation Coefficient', color=COLORS[5])
    
    ax = axes[1]
    
    year_make = df.groupby(['Model Year', 'Make']).size().reset_index(name='Count')
    top_makes_list = df['Make'].value_counts().head(6).index.tolist()
    year_make_top = year_make[year_make['Make'].isin(top_makes_list)]
    
    for i, make in enumerate(top_makes_list):
        data = year_make_top[year_make_top['Make'] == make]
        ax.plot(data['Model Year'], data['Count'], 
               linewidth=3, marker='o', markersize=7,
               color=GRADIENT[i+1], label=make,
               markeredgecolor=COLORS[4], markeredgewidth=1.5, alpha=0.9)
    
    ax.set_facecolor('#0F1415')
    ax.set_title('Temporal Evolution by Manufacturer', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Registrations', fontsize=12, color=COLORS[5])
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4], loc='upper left')
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}correlation_nexus.png', dpi=300, facecolor='#0A0F10', edgecolor='none', bbox_inches='tight')
    plt.close()

def executive_synthesis(df):
    fig, axes = plt.subplots(2, 3, figsize=(28, 16))
    fig.patch.set_facecolor('#0A0F10')
    
    ax = axes[0, 0]
    growth_rate = df.groupby('Model Year').size().pct_change() * 100
    ax.plot(growth_rate.index, growth_rate.values, linewidth=3, color=COLORS[5],
           marker='o', markersize=8, markerfacecolor=COLORS[4], 
           markeredgecolor=COLORS[5], markeredgewidth=2)
    ax.axhline(y=0, color=COLORS[2], linestyle='--', linewidth=2, alpha=0.5)
    ax.fill_between(growth_rate.index, growth_rate.values, 0, 
                    where=(growth_rate.values > 0), alpha=0.3, color=COLORS[5])
    ax.fill_between(growth_rate.index, growth_rate.values, 0, 
                    where=(growth_rate.values < 0), alpha=0.3, color=COLORS[2])
    ax.set_facecolor('#0F1415')
    ax.set_title('YoY Growth Dynamics', fontsize=14, color=COLORS[5], weight='bold', pad=15)
    ax.set_xlabel('Model Year', fontsize=11, color=COLORS[5])
    ax.set_ylabel('Growth Rate (%)', fontsize=11, color=COLORS[5])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[0, 1]
    tesla_vs_others = df.copy()
    tesla_vs_others['Category'] = tesla_vs_others['Make'].apply(lambda x: 'Tesla' if x == 'TESLA' else 'Others')
    market_share = tesla_vs_others.groupby(['Model Year', 'Category']).size().unstack(fill_value=0)
    market_share_pct = market_share.div(market_share.sum(axis=1), axis=0) * 100
    
    ax.plot(market_share_pct.index, market_share_pct['Tesla'], linewidth=4, 
           color=COLORS[5], marker='D', markersize=8, label='Tesla',
           markeredgecolor=COLORS[4], markeredgewidth=2)
    ax.plot(market_share_pct.index, market_share_pct['Others'], linewidth=4,
           color=COLORS[2], marker='s', markersize=8, label='Others',
           markeredgecolor=COLORS[4], markeredgewidth=2)
    ax.set_facecolor('#0F1415')
    ax.set_title('Market Dominance Shift', fontsize=14, color=COLORS[5], weight='bold', pad=15)
    ax.set_xlabel('Model Year', fontsize=11, color=COLORS[5])
    ax.set_ylabel('Market Share (%)', fontsize=11, color=COLORS[5])
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[0, 2]
    model_diversity = df.groupby('Model Year')['Make_Model'].nunique()
    ax.bar(model_diversity.index, model_diversity.values, 
          color=COLORS[4], alpha=0.8, edgecolor=COLORS[5], linewidth=2)
    
    z = np.polyfit(model_diversity.index, model_diversity.values, 2)
    p = np.poly1d(z)
    ax.plot(model_diversity.index, p(model_diversity.index), 
           linewidth=3, color=COLORS[5], linestyle='--', label='Trend')
    
    ax.set_facecolor('#0F1415')
    ax.set_title('Model Diversity Index', fontsize=14, color=COLORS[5], weight='bold', pad=15)
    ax.set_xlabel('Model Year', fontsize=11, color=COLORS[5])
    ax.set_ylabel('Unique Models', fontsize=11, color=COLORS[5])
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 0]
    cumulative = df.groupby('Model Year').size().cumsum()
    ax.fill_between(cumulative.index, cumulative.values, alpha=0.4, color=COLORS[4])
    ax.plot(cumulative.index, cumulative.values, linewidth=3, color=COLORS[5],
           marker='o', markersize=6, markeredgecolor=COLORS[4], markeredgewidth=1.5)
    ax.set_facecolor('#0F1415')
    ax.set_title('Cumulative Fleet Size', fontsize=14, color=COLORS[5], weight='bold', pad=15)
    ax.set_xlabel('Model Year', fontsize=11, color=COLORS[5])
    ax.set_ylabel('Total EVs', fontsize=11, color=COLORS[5])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    for i, val in enumerate(cumulative.values[::3]):
        if i*3 < len(cumulative.index):
            ax.text(cumulative.index[i*3], val, f'{val:,.0f}', 
                   ha='center', va='bottom', color=COLORS[5], fontsize=8)
    
    ax = axes[1, 1]
    phev_bev_ratio = df.groupby('Model Year')['EV_Binary'].apply(lambda x: x.sum() / len(x))
    
    ax.fill_between(phev_bev_ratio.index, phev_bev_ratio.values, alpha=0.3, color=COLORS[5])
    ax.plot(phev_bev_ratio.index, phev_bev_ratio.values, linewidth=3, 
           color=COLORS[5], marker='o', markersize=7,
           markeredgecolor=COLORS[4], markeredgewidth=2)
    ax.set_facecolor('#0F1415')
    ax.set_title('BEV Penetration Rate', fontsize=14, color=COLORS[5], weight='bold', pad=15)
    ax.set_xlabel('Model Year', fontsize=11, color=COLORS[5])
    ax.set_ylabel('BEV Ratio', fontsize=11, color=COLORS[5])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 2]
    top_models_recent = df[df['Model Year'] >= 2022]['Make_Model'].value_counts().head(12)
    
    y_pos = np.arange(len(top_models_recent))
    bars = ax.barh(y_pos, top_models_recent.values,
                   color=[GRADIENT[i % len(GRADIENT)] for i in range(len(top_models_recent))],
                   edgecolor=COLORS[4], linewidth=1.5, alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_models_recent.index, fontsize=9, color=COLORS[5])
    ax.set_xlabel('Registrations', fontsize=11, color=COLORS[5])
    ax.set_title('Current Leaders (2022+)', fontsize=14, color=COLORS[5], weight='bold', pad=15)
    ax.set_facecolor('#0F1415')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
    ax.tick_params(colors=COLORS[5])
    
    for i, (bar, val) in enumerate(zip(bars, top_models_recent.values)):
        ax.text(val, i, f' {val:,}', va='center', ha='left', 
               color=COLORS[5], fontsize=8)
    
    plt.suptitle('EXECUTIVE INTELLIGENCE DASHBOARD', 
                fontsize=20, color=COLORS[5], weight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}executive_synthesis.png', dpi=300, facecolor='#0A0F10', edgecolor='none', bbox_inches='tight')
    plt.close()

def neural_insights(df):
    total = len(df)
    bev_count = df['EV_Binary'].sum()
    cafv_count = df['CAFV_Binary'].sum()
    
    unique_makes = df['Make'].nunique()
    unique_models = df['Make_Model'].nunique()
    
    avg_year = df['Model Year'].mean()
    median_year = df['Model Year'].median()
    
    top_county = df['County'].value_counts().index[0]
    top_city = df['City'].value_counts().index[0]
    top_make = df['Make'].value_counts().index[0]
    
    recent_growth = df[df['Model Year'] >= 2020].groupby('Model Year').size()
    if len(recent_growth) > 1:
        cagr = ((recent_growth.iloc[-1] / recent_growth.iloc[0]) ** (1/(len(recent_growth)-1)) - 1) * 100
    else:
        cagr = 0
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#0A0F10')
    ax.axis('off')
    
    title_text = 'QUANTUM INTELLIGENCE SYNTHESIS'
    ax.text(0.5, 0.95, title_text, fontsize=28, color=COLORS[5], 
           weight='bold', ha='center', va='top')
    
    metrics = [
        ('TOTAL FLEET SIZE', f'{total:,}', 0.15),
        ('BEV DOMINANCE', f'{bev_count/total*100:.1f}%', 0.35),
        ('CAFV ELIGIBLE', f'{cafv_count/total*100:.1f}%', 0.55),
        ('RECENT CAGR', f'{cagr:.1f}%', 0.75),
    ]
    
    for label, value, x_pos in metrics:
        ax.text(x_pos, 0.80, label, fontsize=12, color=COLORS[4],
               ha='center', va='bottom', weight='bold')
        ax.text(x_pos, 0.72, value, fontsize=24, color=COLORS[5],
               ha='center', va='top', weight='bold')
    
    insights = [
        f'Market Diversity: {unique_makes} manufacturers offering {unique_models} unique models',
        f'Geographic Concentration: {top_county} County leads ({df["County"].value_counts().iloc[0]:,} units)',
        f'Urban Epicenter: {top_city} dominates city-level registrations',
        f'Market Leader: {top_make} commands {df["Make"].value_counts().iloc[0]/total*100:.1f}% market share',
        f'Fleet Maturity: Average model year {avg_year:.1f} (Median: {median_year:.0f})',
        f'Technology Shift: BEV adoption accelerating post-2020',
    ]
    
    y_start = 0.55
    for i, insight in enumerate(insights):
        y_pos = y_start - (i * 0.08)
        
        bullet_x = 0.08
        ax.plot([bullet_x-0.01, bullet_x], [y_pos, y_pos], linewidth=3, 
               color=GRADIENT[i % len(GRADIENT)], marker='o', markersize=10,
               markerfacecolor=GRADIENT[i % len(GRADIENT)], markeredgecolor=COLORS[4],
               markeredgewidth=2)
        
        ax.text(bullet_x + 0.03, y_pos, insight, fontsize=13, color=COLORS[5],
               ha='left', va='center', style='italic')
    
    footer_text = 'Generated by Advanced Analytics Engine  •  Temporal Analysis Framework  •  Predictive Intelligence System'
    ax.text(0.5, 0.02, footer_text, fontsize=10, color=COLORS[4],
           ha='center', va='bottom', alpha=0.7)
    
    plt.savefig(f'{OUTPUT_PATH}neural_insights.png', dpi=300, facecolor='#0A0F10', edgecolor='none', bbox_inches='tight')
    plt.close()

def main():
    print("Initializing quantum analysis framework...")
    
    df = pd.read_csv('/Users/cazandraaporbo/Desktop/mygit/Electric_Vehicles/Electric_Vehicle_Population_Data.csv')
    
    print(f"Dataset loaded: {len(df):,} records across {len(df.columns)} dimensions")
    
    print("Executing data purification protocols...")
    df = quantum_clean(df)
    
    print("Generating temporal cascade visualization...")
    temporal_cascade(df)
    
    print("Computing dimensional reduction manifolds...")
    dimensional_reduction(df)
    
    print("Constructing statistical manifolds...")
    statistical_manifold(df)
    
    print("Rendering advanced distribution matrices...")
    advanced_distribution(df)
    
    print("Mapping correlation nexus structures...")
    correlation_nexus(df)
    
    print("Synthesizing executive intelligence dashboard...")
    executive_synthesis(df)
    
    print("Extracting neural insights...")
    neural_insights(df)
    
    print(f"\nAnalysis complete. 7 visualization suites deployed to: {OUTPUT_PATH}")
    print("Quantum analysis framework terminated successfully.")

if __name__ == '__main__':
    main()
