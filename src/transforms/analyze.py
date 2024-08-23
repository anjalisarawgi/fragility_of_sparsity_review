# import pandas as pd
# import plotly.graph_objects as go
# import us
# from plotly.offline import iplot
# import os



# # Load the data
# file_path = 'Data/processed/communities_and_crime.csv'
# data = pd.read_csv(file_path)


# output_dir = 'reports/figures'
# os.makedirs(output_dir, exist_ok=True)

# # Convert FIPS codes to state abbreviations using the 'us' package
# def fips_to_state_abbr(fips_code):
#     try:
#         return us.states.lookup(str(fips_code).zfill(2)).abbr
#     except:
#         return None

# print("shape of data: ", data.shape)
# data['state_abbr'] = data['state'].apply(fips_to_state_abbr)
# print("shape of data: ", data.shape)
# print("data.head(): ", data['state_abbr'].head())
# # convert state_abbr to categorical
# data['state_abbr'] = data['state_abbr'].astype('category')

# missing_states = data[data['state_abbr'].isnull()] # Find missing state abbreviations
# # print(missing_states[['state', 'ViolentCrimesPerPop']])

# # Manual mapping for missing FIPS codes (if needed)
# manual_fips_map = {
#     '30': 'MT',  # Montana
#     '31': 'NE',  # Nebraska
#     '17': 'IL',  # Illinois
#     '26': 'MI',  # Michigan
#     '15': 'HI'   # Hawaii
# }

# data['state_abbr'] = data.apply(
#     lambda row: manual_fips_map.get(row['state'], row['state_abbr']),
#     axis=1
# )
# # save the data
# data.to_csv('Data/processed/communities_and_crime_2.csv', index=False)

# """
# 1. Check which states have data available
# """
# data['data_flag'] = data['state_abbr'].apply(lambda x: 1 if pd.notnull(x) else 0)

# test_data = dict(
#     type='choropleth',
#     colorscale='Blues',
#     locations=data['state_abbr'],
#     locationmode='USA-states',
#     z=data['data_flag'],
#     colorbar={'title': 'Data Present'}
# )

# test_layout = dict(
#     title='Test Map: States with Data',
#     geo=dict(
#         scope='usa',
#         projection=dict(type='albers usa'),
#         showlakes=True,
#         lakecolor='rgb(85,173,240)'
#     )
# )

# test_fig = go.Figure(data=[test_data], layout=test_layout)
# # iplot(test_fig, validate=False)
# test_fig.write_html(f'{output_dir}/test_map.html')
# test_fig.write_image(f'{output_dir}/test_map.png')


# """
# 2. Interactive map to visualize ViolentCrimesPerPop
# """
# data1 = dict(
#     type='choropleth',
#     colorscale='Viridis',
#     autocolorscale=False,
#     locations=data['state_abbr'],
#     locationmode='USA-states',
#     z=data['ViolentCrimesPerPop'].astype(float),
#     colorbar={'title': 'Violent Crimes (Per-100K-Pop)'}
# )

# layout1 = dict(
#     title='Aggregate view of Violent Crimes Per 100K Population',
#     geo=dict(
#         scope='usa',
#         projection=dict(type='albers usa'),
#         showlakes=True,
#         lakecolor='rgb(85,173,240)'
#     )
# )

# fig1 = go.Figure(data=[data1], layout=layout1)
# # iplot(fig1, validate=False)
# fig1.write_html(f'{output_dir}/violent_crimes_map.html')
# fig1.write_image(f'{output_dir}/violent_crimes_map.png')


