from plotly.offline import iplot
import os
import pandas as pd
import plotly.graph_objects as go


data = pd.read_csv('Data/communities_and_crime/processed/communities_and_crime.csv')
output_dir = 'reports/figures'


# Dataset = Communities and Crime
"""
1. Check which states have data available
"""
data['data_flag'] = data['state_abbr'].apply(lambda x: 1 if pd.notnull(x) else 0)

test_data = dict(
    type='choropleth',
    colorscale='Blues',
    locations=data['state_abbr'],
    locationmode='USA-states',
    z=data['data_flag'],
    colorbar={'title': 'Data Present'}
)

test_layout = dict(
    title='Test Map: States with Data',
    geo=dict(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(85,173,240)'
    )
)

test_fig = go.Figure(data=[test_data], layout=test_layout)
# iplot(test_fig, validate=False)
test_fig.write_html(f'{output_dir}/test_map.html')
test_fig.write_image(f'{output_dir}/test_map.png')


"""
2. Aggregate view of Violent Crimes Per 100K Population
"""
data1 = dict(
    type='choropleth',
    colorscale='Viridis',
    autocolorscale=False,
    locations=data['state_abbr'],
    locationmode='USA-states',
    z=data['ViolentCrimesPerPop'].astype(float),
    colorbar={'title': 'Violent Crimes (Per-100K-Pop)'}
)

layout1 = dict(
    title='Aggregate view of Violent Crimes Per 100K Population',
    geo=dict(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(85,173,240)'
    )
)

fig1 = go.Figure(data=[data1], layout=layout1)
# iplot(fig1, validate=False)
fig1.write_html(f'{output_dir}/violent_crimes_map.html')
fig1.write_image(f'{output_dir}/violent_crimes_map.png')


