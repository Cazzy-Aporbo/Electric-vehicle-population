import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#172226', '#000000', '#5F5CA4', '#476F75', '#597C82', '#B2E4D9']
GRADIENT = ['#172226', '#2A3A3E', '#3D5156', '#476F75', '#597C82', '#6B8A8F', '#8BA8A8', '#A9C3C1', '#B2E4D9']
OUTPUT_PATH = '/Users/cazandraaporbo/Desktop/mygit/Electric_Vehicles/'

plt.style.use('dark_background')

class TemporalSpatialIndex:
    def __init__(self, df):
        self.raw = df.copy()
        self.transformed = None
        self.synthesis()
        
    def synthesis(self):
        county_encode = pd.factorize(self.raw['County'])[0]
        make_encode = pd.factorize(self.raw['Make'])[0]
        
        temporal_weight = (self.raw['Model Year'] - self.raw['Model Year'].min()) / (self.raw['Model Year'].max() - self.raw['Model Year'].min())
        spatial_entropy = county_encode / county_encode.max()
        market_density = make_encode / make_encode.max()
        
        composite_index = (temporal_weight * 0.4 + spatial_entropy * 0.3 + market_density * 0.3)
        
        self.raw['TSI_Score'] = composite_index
        self.raw['TSI_Quartile'] = pd.qcut(composite_index, q=4, labels=['Genesis', 'Expansion', 'Maturation', 'Singularity'])
        
        adoption_velocity = self.raw.groupby(['County', 'Model Year']).size().reset_index(name='velocity')
        adoption_velocity['velocity_delta'] = adoption_velocity.groupby('County')['velocity'].diff().fillna(0)
        
        self.raw = self.raw.merge(
            adoption_velocity[['County', 'Model Year', 'velocity_delta']], 
            on=['County', 'Model Year'], 
            how='left'
        )
        
        make_year_matrix = self.raw.groupby(['Make', 'Model Year']).size().unstack(fill_value=0)
        correlation_matrix = make_year_matrix.T.corr()
        
        self.correlation_network = correlation_matrix
        self.temporal_clusters = self._build_clusters()
        
        self.transformed = self.raw
        
    def _build_clusters(self):
        feature_matrix = self.raw.groupby('Make').agg({
            'Model Year': ['mean', 'std'],
            'EV_Binary': 'mean',
            'TSI_Score': 'mean',
            'VIN (1-10)': 'count'
        }).fillna(0)
        
        feature_matrix.columns = ['_'.join(col).strip() for col in feature_matrix.columns.values]
        feature_matrix = feature_matrix[feature_matrix['VIN (1-10)_count'] >= 100]
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        linkage_matrix = linkage(scaled_features, method='ward')
        
        return {
            'linkage': linkage_matrix,
            'labels': feature_matrix.index.tolist(),
            'features': feature_matrix
        }

def load_transform():
    df = pd.read_csv('/Users/cazandraaporbo/Desktop/mygit/Electric_Vehicles/Electric_Vehicle_Population_Data.csv')
    
    df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')
    df = df[(df['Model Year'] >= 1999) & (df['Model Year'] <= 2026)]
    df = df[df['State'].notna()]
    
    df['EV_Binary'] = (df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)').astype(int)
    df['CAFV_Binary'] = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].str.contains('Eligible', na=False).astype(int)
    df['Make_Model'] = df['Make'] + ' ' + df['Model']
    
    tsi = TemporalSpatialIndex(df)
    return tsi.transformed, tsi

def hierarchical_taxonomy(tsi):
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.patch.set_facecolor('#0A0F10')
    
    ax = axes[0]
    dendrogram(
        tsi.temporal_clusters['linkage'],
        labels=tsi.temporal_clusters['labels'],
        orientation='left',
        color_threshold=0,
        above_threshold_color=COLORS[4],
        ax=ax
    )
    ax.set_facecolor('#0F1415')
    ax.set_title('Hierarchical Manufacturer Taxonomy', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.tick_params(colors=COLORS[5], labelsize=9)
    ax.spines['top'].set_color(COLORS[4])
    ax.spines['bottom'].set_color(COLORS[4])
    ax.spines['left'].set_color(COLORS[4])
    ax.spines['right'].set_color(COLORS[4])
    
    for i, line in enumerate(ax.get_lines()):
        line.set_color(GRADIENT[i % len(GRADIENT)])
        line.set_linewidth(2)
    
    ax = axes[1]
    
    top_makes = tsi.transformed['Make'].value_counts().head(12).index
    filtered_corr = tsi.correlation_network.loc[top_makes, top_makes]
    
    mask = np.triu(np.ones_like(filtered_corr, dtype=bool), k=1)
    filtered_corr_masked = filtered_corr.copy()
    filtered_corr_masked[mask] = np.nan
    
    im = ax.imshow(filtered_corr_masked, cmap='twilight', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(top_makes)))
    ax.set_yticks(np.arange(len(top_makes)))
    ax.set_xticklabels(top_makes, rotation=45, ha='right', fontsize=9, color=COLORS[5])
    ax.set_yticklabels(top_makes, fontsize=9, color=COLORS[5])
    
    for i in range(len(top_makes)):
        for j in range(i):
            val = filtered_corr_masked.iloc[i, j]
            if not np.isnan(val):
                text_color = COLORS[5] if abs(val) < 0.5 else COLORS[1]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=text_color, fontsize=8, weight='bold')
    
    ax.set_facecolor('#0F1415')
    ax.set_title('Temporal Correlation Matrix', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', color=COLORS[5], fontsize=11)
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}hierarchical_taxonomy.png', dpi=300, facecolor='#0A0F10', edgecolor='none', bbox_inches='tight')
    plt.close()

def phase_space_trajectories(df):
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    fig.patch.set_facecolor('#0A0F10')
    
    ax = axes[0, 0]
    
    yearly_data = df.groupby('Model Year').agg({
        'VIN (1-10)': 'count',
        'EV_Binary': 'mean',
        'CAFV_Binary': 'mean'
    }).reset_index()
    
    points = yearly_data[['VIN (1-10)', 'EV_Binary']].values
    segments = np.array([points[i:i+2] for i in range(len(points)-1)])
    
    norm = plt.Normalize(yearly_data['Model Year'].min(), yearly_data['Model Year'].max())
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidths=3, alpha=0.8)
    lc.set_array(yearly_data['Model Year'].values[:-1])
    
    line = ax.add_collection(lc)
    
    scatter = ax.scatter(yearly_data['VIN (1-10)'], yearly_data['EV_Binary'],
               c=yearly_data['Model Year'], cmap='viridis', s=200, 
               edgecolors=COLORS[4], linewidths=2, zorder=5)
    
    for idx, row in yearly_data.iterrows():
        if idx % 3 == 0:
            ax.annotate(f"{int(row['Model Year'])}", 
                       (row['VIN (1-10)'], row['EV_Binary']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, color=COLORS[5], weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#0A0F10', 
                                edgecolor=COLORS[4], alpha=0.8))
    
    ax.set_facecolor('#0F1415')
    ax.set_xlabel('Total Registrations', fontsize=12, color=COLORS[5])
    ax.set_ylabel('BEV Ratio', fontsize=12, color=COLORS[5])
    ax.set_title('Phase Space Evolution', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Model Year', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[0, 1]
    
    top_makes = df['Make'].value_counts().head(8).index
    violin_data = []
    labels = []
    
    for make in top_makes:
        years = df[df['Make'] == make]['Model Year'].values
        if len(years) > 0:
            violin_data.append(years)
            labels.append(make)
    
    parts = ax.violinplot(violin_data, positions=range(len(labels)), 
                          widths=0.7, showmeans=True, showextrema=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(GRADIENT[i % len(GRADIENT)])
        pc.set_edgecolor(COLORS[4])
        pc.set_alpha(0.7)
        pc.set_linewidth(2)
    
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        vp.set_edgecolor(COLORS[5])
        vp.set_linewidth(2)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10, color=COLORS[5])
    ax.set_ylabel('Model Year Distribution', fontsize=12, color=COLORS[5])
    ax.set_title('Manufacturer Temporal Distribution', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_facecolor('#0F1415')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 0]
    
    tsi_year = df.groupby(['TSI_Quartile', 'Model Year']).size().unstack(fill_value=0)
    
    bottom = np.zeros(len(tsi_year.columns))
    for i, quartile in enumerate(tsi_year.index):
        ax.bar(tsi_year.columns, tsi_year.loc[quartile], bottom=bottom,
              label=quartile, color=GRADIENT[i*2], alpha=0.85,
              edgecolor=COLORS[4], linewidth=1.5)
        bottom += tsi_year.loc[quartile].values
    
    ax.set_facecolor('#0F1415')
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Registration Count', fontsize=12, color=COLORS[5])
    ax.set_title('TSI Quartile Stratification', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 1]
    
    county_velocity = df.groupby('County')['velocity_delta'].mean().sort_values(ascending=False).head(20)
    
    colors_map = [GRADIENT[i % len(GRADIENT)] for i in range(len(county_velocity))]
    bars = ax.barh(range(len(county_velocity)), county_velocity.values, 
                   color=colors_map, edgecolor=COLORS[4], linewidth=1.5, alpha=0.85)
    
    ax.set_yticks(range(len(county_velocity)))
    ax.set_yticklabels(county_velocity.index, fontsize=9, color=COLORS[5])
    ax.set_xlabel('Adoption Velocity Delta', fontsize=12, color=COLORS[5])
    ax.set_title('Geographic Acceleration Index', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_facecolor('#0F1415')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
    ax.tick_params(colors=COLORS[5])
    ax.axvline(x=0, color=COLORS[2], linewidth=2, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}phase_space_trajectories.png', dpi=300, facecolor='#0A0F10', edgecolor='none', bbox_inches='tight')
    plt.close()

def market_topology(df):
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor('#0A0F10')
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    ax = fig.add_subplot(gs[0:2, 0:2])
    
    model_counts = df['Make_Model'].value_counts().head(30)
    
    sizes = model_counts.values
    labels = model_counts.index
    
    total = sum(sizes)
    sizes_normalized = [s/total for s in sizes]
    
    x = 0
    y = 0
    w = 100
    h = 100
    
    rectangles = []
    
    def squarify_layout(sizes, x, y, w, h):
        if not sizes:
            return []
        
        if len(sizes) == 1:
            return [(x, y, w, h, sizes[0])]
        
        total = sum(sizes)
        mid = len(sizes) // 2
        
        left_sum = sum(sizes[:mid])
        right_sum = sum(sizes[mid:])
        
        if w >= h:
            w_left = w * (left_sum / total)
            left_rects = squarify_layout(sizes[:mid], x, y, w_left, h)
            right_rects = squarify_layout(sizes[mid:], x + w_left, y, w - w_left, h)
        else:
            h_top = h * (left_sum / total)
            left_rects = squarify_layout(sizes[:mid], x, y, w, h_top)
            right_rects = squarify_layout(sizes[mid:], x, y + h_top, w, h - h_top)
        
        return left_rects + right_rects
    
    layout = squarify_layout(sizes_normalized, x, y, w, h)
    
    for i, (rx, ry, rw, rh, size) in enumerate(layout[:len(labels)]):
        color = GRADIENT[i % len(GRADIENT)]
        rect = Rectangle((rx, ry), rw, rh, facecolor=color, 
                        edgecolor=COLORS[0], linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        
        if rw > 8 and rh > 8:
            label_text = labels[i].split()[1] if len(labels[i].split()) > 1 else labels[i]
            ax.text(rx + rw/2, ry + rh/2, f'{label_text}\n{model_counts.values[i]:,}',
                   ha='center', va='center', fontsize=8, color=COLORS[5],
                   weight='bold', wrap=True)
    
    ax.set_xlim(x, x + w)
    ax.set_ylim(y, y + h)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Market Topology: Model Distribution', fontsize=18, color=COLORS[5], 
                pad=20, weight='bold')
    ax.set_facecolor('#0F1415')
    
    ax = fig.add_subplot(gs[0, 2])
    
    type_counts = df['Electric Vehicle Type'].value_counts()
    explode = [0.05] * len(type_counts)
    
    wedges, texts, autotexts = ax.pie(type_counts.values, labels=type_counts.index,
                                        autopct='%1.1f%%', startangle=90, explode=explode,
                                        colors=[COLORS[2], COLORS[4]],
                                        wedgeprops={'linewidth': 3, 'edgecolor': COLORS[0]},
                                        textprops={'color': COLORS[5], 'fontsize': 10})
    
    for autotext in autotexts:
        autotext.set_color(COLORS[5])
        autotext.set_weight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title('Technology Split', fontsize=14, color=COLORS[5], pad=15, weight='bold')
    
    ax = fig.add_subplot(gs[1, 2])
    
    cafv_counts = df['CAFV_Binary'].value_counts()
    colors_cafv = [COLORS[2], COLORS[4]]
    
    wedges, texts, autotexts = ax.pie(cafv_counts.values, 
                                        labels=['Not Eligible', 'Eligible'],
                                        autopct='%1.1f%%', startangle=90,
                                        colors=colors_cafv, explode=[0.05, 0.05],
                                        wedgeprops={'linewidth': 3, 'edgecolor': COLORS[0]},
                                        textprops={'color': COLORS[5], 'fontsize': 10})
    
    for autotext in autotexts:
        autotext.set_color(COLORS[5])
        autotext.set_weight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title('CAFV Status', fontsize=14, color=COLORS[5], pad=15, weight='bold')
    
    ax = fig.add_subplot(gs[2, :])
    
    city_make = df.groupby(['City', 'Make']).size().reset_index(name='count')
    top_cities = df['City'].value_counts().head(15).index
    top_makes_for_sankey = df['Make'].value_counts().head(6).index
    
    city_make_filtered = city_make[
        (city_make['City'].isin(top_cities)) & 
        (city_make['Make'].isin(top_makes_for_sankey))
    ].sort_values('count', ascending=False).head(40)
    
    city_totals = city_make_filtered.groupby('City')['count'].sum().sort_values(ascending=False)
    make_totals = city_make_filtered.groupby('Make')['count'].sum().sort_values(ascending=False)
    
    y_cities = {city: i * 1.5 for i, city in enumerate(city_totals.index)}
    y_makes = {make: i * 1.5 for i, make in enumerate(make_totals.index)}
    
    x_city = 0
    x_make = 10
    
    for city in city_totals.index:
        ax.add_patch(Rectangle((x_city - 0.3, y_cities[city] - 0.5), 0.6, 1,
                              facecolor=COLORS[4], edgecolor=COLORS[5], linewidth=2))
        ax.text(x_city - 1, y_cities[city], city, ha='right', va='center',
               fontsize=9, color=COLORS[5])
    
    for make in make_totals.index:
        ax.add_patch(Rectangle((x_make - 0.3, y_makes[make] - 0.5), 0.6, 1,
                              facecolor=COLORS[4], edgecolor=COLORS[5], linewidth=2))
        ax.text(x_make + 1, y_makes[make], make, ha='left', va='center',
               fontsize=9, color=COLORS[5])
    
    max_flow = city_make_filtered['count'].max()
    
    for idx, row in city_make_filtered.iterrows():
        city = row['City']
        make = row['Make']
        count = row['count']
        
        if city in y_cities and make in y_makes:
            y1 = y_cities[city]
            y2 = y_makes[make]
            
            alpha = 0.3 + 0.5 * (count / max_flow)
            color_idx = list(make_totals.index).index(make) % len(GRADIENT)
            
            ax.plot([x_city + 0.3, x_make - 0.3], [y1, y2],
                   linewidth=count/max_flow * 15, alpha=alpha,
                   color=GRADIENT[color_idx], solid_capstyle='round')
    
    ax.set_xlim(-3, 13)
    ax.set_ylim(-1, max(max(y_cities.values()), max(y_makes.values())) + 1)
    ax.axis('off')
    ax.set_title('City-Manufacturer Flow Network', fontsize=16, color=COLORS[5], 
                pad=20, weight='bold')
    ax.set_facecolor('#0F1415')
    
    plt.savefig(f'{OUTPUT_PATH}market_topology.png', dpi=300, facecolor='#0A0F10', 
                edgecolor='none', bbox_inches='tight')
    plt.close()

def temporal_decomposition(df):
    fig, axes = plt.subplots(3, 2, figsize=(24, 18))
    fig.patch.set_facecolor('#0A0F10')
    
    ax = axes[0, 0]
    
    yearly_total = df.groupby('Model Year').size()
    
    ax.fill_between(yearly_total.index, yearly_total.values, alpha=0.3, color=COLORS[4])
    ax.plot(yearly_total.index, yearly_total.values, linewidth=3, color=COLORS[5],
           marker='o', markersize=8, markeredgecolor=COLORS[4], markeredgewidth=2)
    
    window = 3
    if len(yearly_total) >= window:
        trend = yearly_total.rolling(window=window, center=True).mean()
        ax.plot(trend.index, trend.values, linewidth=4, color=COLORS[2],
               linestyle='--', alpha=0.8, label=f'{window}-Year Trend')
    
    ax.set_facecolor('#0F1415')
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Registrations', fontsize=12, color=COLORS[5])
    ax.set_title('Temporal Series with Trend', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[0, 1]
    
    seasonal = df.groupby(['Model Year', 'TSI_Quartile']).size().unstack(fill_value=0)
    
    for i, quartile in enumerate(seasonal.columns):
        ax.plot(seasonal.index, seasonal[quartile], linewidth=3,
               marker='o', markersize=6, label=quartile,
               color=GRADIENT[i*2], markeredgecolor=COLORS[4], markeredgewidth=1.5)
    
    ax.set_facecolor('#0F1415')
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Count', fontsize=12, color=COLORS[5])
    ax.set_title('Seasonal Components by TSI', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 0]
    
    top_models = df['Make_Model'].value_counts().head(15).index
    model_year_matrix = df[df['Make_Model'].isin(top_models)].groupby(['Make_Model', 'Model Year']).size().unstack(fill_value=0)
    
    im = ax.imshow(model_year_matrix.values, aspect='auto', cmap='plasma', interpolation='nearest')
    
    ax.set_yticks(range(len(model_year_matrix.index)))
    ax.set_yticklabels(model_year_matrix.index, fontsize=8, color=COLORS[5])
    ax.set_xticks(range(0, len(model_year_matrix.columns), 2))
    ax.set_xticklabels(model_year_matrix.columns[::2], fontsize=9, color=COLORS[5])
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_title('Model Temporal Heatmap', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_facecolor('#0F1415')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Registrations', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[1, 1]
    
    county_year = df.groupby(['County', 'Model Year']).size().reset_index(name='count')
    top_counties_list = df['County'].value_counts().head(10).index
    
    for i, county in enumerate(top_counties_list):
        county_data = county_year[county_year['County'] == county]
        ax.plot(county_data['Model Year'], county_data['count'],
               linewidth=2.5, marker='o', markersize=5, label=county,
               color=GRADIENT[i % len(GRADIENT)], alpha=0.8, markeredgecolor=COLORS[4], markeredgewidth=1)
    
    ax.set_facecolor('#0F1415')
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Registrations', fontsize=12, color=COLORS[5])
    ax.set_title('County Evolution Trajectories', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4], 
             fontsize=8, ncol=2)
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[2, 0]
    
    bev_phev_year = df.groupby(['Model Year', 'Electric Vehicle Type']).size().unstack(fill_value=0)
    
    x = np.arange(len(bev_phev_year.index))
    width = 0.4
    
    if 'Battery Electric Vehicle (BEV)' in bev_phev_year.columns:
        ax.bar(x - width/2, bev_phev_year['Battery Electric Vehicle (BEV)'], 
              width, label='BEV', color=COLORS[4], alpha=0.85,
              edgecolor=COLORS[5], linewidth=2)
    
    if 'Plug-in Hybrid Electric Vehicle (PHEV)' in bev_phev_year.columns:
        ax.bar(x + width/2, bev_phev_year['Plug-in Hybrid Electric Vehicle (PHEV)'],
              width, label='PHEV', color=COLORS[2], alpha=0.85,
              edgecolor=COLORS[5], linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(bev_phev_year.index, rotation=45, ha='right', fontsize=9, color=COLORS[5])
    ax.set_ylabel('Registrations', fontsize=12, color=COLORS[5])
    ax.set_title('Technology Type Comparison', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.set_facecolor('#0F1415')
    ax.legend(framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[2, 1]
    
    residual = yearly_total.copy()
    if len(yearly_total) >= window:
        trend = yearly_total.rolling(window=window, center=True).mean()
        residual = yearly_total - trend
        residual = residual.dropna()
    
    bars = ax.bar(residual.index, residual.values, 
                  color=[COLORS[4] if x >= 0 else COLORS[2] for x in residual.values],
                  alpha=0.85, edgecolor=COLORS[5], linewidth=1.5)
    
    ax.axhline(y=0, color=COLORS[5], linewidth=2, linestyle='-')
    ax.set_facecolor('#0F1415')
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Residual', fontsize=12, color=COLORS[5])
    ax.set_title('Residual Analysis', fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}temporal_decomposition.png', dpi=300, facecolor='#0A0F10', 
                edgecolor='none', bbox_inches='tight')
    plt.close()

def multidimensional_scatter(df):
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    fig.patch.set_facecolor('#0A0F10')
    
    ax = axes[0, 0]
    
    county_stats = df.groupby('County').agg({
        'VIN (1-10)': 'count',
        'EV_Binary': 'mean',
        'CAFV_Binary': 'mean',
        'TSI_Score': 'mean'
    }).reset_index()
    county_stats = county_stats[county_stats['VIN (1-10)'] >= 100]
    
    scatter = ax.scatter(county_stats['VIN (1-10)'], 
                        county_stats['EV_Binary'] * 100,
                        s=county_stats['CAFV_Binary'] * 1000 + 100,
                        c=county_stats['TSI_Score'],
                        cmap='cool', alpha=0.7,
                        edgecolors=COLORS[4], linewidths=2)
    
    top_counties = county_stats.nlargest(8, 'VIN (1-10)')
    for idx, row in top_counties.iterrows():
        ax.annotate(row['County'], 
                   (row['VIN (1-10)'], row['EV_Binary'] * 100),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=8, color=COLORS[5],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#0A0F10',
                            edgecolor=COLORS[4], alpha=0.9))
    
    ax.set_facecolor('#0F1415')
    ax.set_xlabel('Total Registrations', fontsize=12, color=COLORS[5])
    ax.set_ylabel('BEV Percentage', fontsize=12, color=COLORS[5])
    ax.set_title('County Multidimensional Analysis', fontsize=16, color=COLORS[5], 
                pad=20, weight='bold')
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('TSI Score', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    size_legend = [
        ax.scatter([], [], s=100, c='none', edgecolors=COLORS[5], linewidths=2, label='Low CAFV'),
        ax.scatter([], [], s=500, c='none', edgecolors=COLORS[5], linewidths=2, label='Mid CAFV'),
        ax.scatter([], [], s=900, c='none', edgecolors=COLORS[5], linewidths=2, label='High CAFV')
    ]
    ax.legend(handles=size_legend, framealpha=0.9, facecolor='#0A0F10', 
             edgecolor=COLORS[4], loc='lower right', fontsize=9)
    
    ax = axes[0, 1]
    
    make_stats = df.groupby('Make').agg({
        'Model Year': 'mean',
        'VIN (1-10)': 'count',
        'EV_Binary': 'mean',
        'TSI_Score': 'std'
    }).reset_index()
    make_stats = make_stats[make_stats['VIN (1-10)'] >= 500]
    
    scatter = ax.scatter(make_stats['Model Year'],
                        make_stats['EV_Binary'] * 100,
                        s=make_stats['VIN (1-10)'] / 10,
                        c=make_stats['TSI_Score'],
                        cmap='viridis', alpha=0.7,
                        edgecolors=COLORS[4], linewidths=2)
    
    for idx, row in make_stats.iterrows():
        if row['VIN (1-10)'] > 5000:
            ax.annotate(row['Make'],
                       (row['Model Year'], row['EV_Binary'] * 100),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color=COLORS[5], weight='bold')
    
    ax.set_facecolor('#0F1415')
    ax.set_xlabel('Average Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('BEV Percentage', fontsize=12, color=COLORS[5])
    ax.set_title('Manufacturer Portfolio Analysis', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('TSI Variance', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[1, 0]
    
    year_velocity = df.groupby('Model Year').agg({
        'velocity_delta': 'mean',
        'TSI_Score': 'mean',
        'VIN (1-10)': 'count'
    }).reset_index()
    
    scatter = ax.scatter(year_velocity['Model Year'],
                        year_velocity['velocity_delta'],
                        s=year_velocity['VIN (1-10)'] / 50,
                        c=year_velocity['TSI_Score'],
                        cmap='plasma', alpha=0.7,
                        edgecolors=COLORS[4], linewidths=2)
    
    ax.axhline(y=0, color=COLORS[2], linewidth=2, linestyle='--', alpha=0.6)
    
    ax.set_facecolor('#0F1415')
    ax.set_xlabel('Model Year', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Average Velocity Delta', fontsize=12, color=COLORS[5])
    ax.set_title('Temporal Velocity Field', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('TSI Score', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[1, 1]
    
    quartile_stats = df.groupby('TSI_Quartile').agg({
        'Model Year': 'mean',
        'VIN (1-10)': 'count',
        'EV_Binary': 'mean',
        'CAFV_Binary': 'mean'
    }).reset_index()
    
    x = np.arange(len(quartile_stats))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, quartile_stats['EV_Binary'] * 100, width,
                   label='BEV %', color=COLORS[4], alpha=0.85,
                   edgecolor=COLORS[5], linewidth=2)
    bars2 = ax.bar(x + width/2, quartile_stats['CAFV_Binary'] * 100, width,
                   label='CAFV %', color=COLORS[2], alpha=0.85,
                   edgecolor=COLORS[5], linewidth=2)
    
    ax2 = ax.twinx()
    line = ax2.plot(x, quartile_stats['VIN (1-10)'], 
                   color=COLORS[5], linewidth=3, marker='D', markersize=10,
                   label='Total Count', markeredgecolor=COLORS[4], 
                   markeredgewidth=2, zorder=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(quartile_stats['TSI_Quartile'], fontsize=11, color=COLORS[5])
    ax.set_ylabel('Percentage', fontsize=12, color=COLORS[5])
    ax.set_title('TSI Quartile Performance', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0F1415')
    ax2.set_facecolor('#0F1415')
    ax.tick_params(colors=COLORS[5])
    ax2.tick_params(colors=COLORS[5])
    ax2.set_ylabel('Total Registrations', fontsize=12, color=COLORS[5])
    
    ax.legend(loc='upper left', framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4])
    ax2.legend(loc='upper right', framealpha=0.9, facecolor='#0A0F10', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}multidimensional_scatter.png', dpi=300, facecolor='#0A0F10',
                edgecolor='none', bbox_inches='tight')
    plt.close()

def main():
    print("Initializing advanced analytics framework")
    
    df, tsi = load_transform()
    print(f"Processed {len(df):,} records with TemporalSpatialIndex transformation")
    
    print("Generating hierarchical taxonomy")
    hierarchical_taxonomy(tsi)
    
    print("Computing phase space trajectories")
    phase_space_trajectories(df)
    
    print("Rendering market topology")
    market_topology(df)
    
    print("Executing temporal decomposition")
    temporal_decomposition(df)
    
    print("Creating multidimensional scatter analysis")
    multidimensional_scatter(df)
    
    print(f"\nAnalysis complete: 5 visualization suites deployed to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
