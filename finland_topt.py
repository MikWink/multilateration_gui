import numpy as np
import numpy.ma as ma # For masked arrays
from PIL import Image
import plotly.graph_objects as go
import plotly.offline as pyo

# Load and downsample
height_map = np.array(Image.open('50n000e_20101117_gmted_mea150_cut.tif'), dtype=np.float64)
print(height_map.shape)
for vals in height_map:
    for val in vals:
        if not 30000 < val < 35000:
            print(val)
height_map -= 30000
# Create meshgrid
x = np.arange(height_map.shape[1])
y = np.arange(height_map.shape[0])
x, y = np.meshgrid(x, y)



# Create Plotly figure
fig = go.Figure(data=[go.Surface(x=y, y=x, z=height_map, colorscale='Viridis', showscale=True, name='Terrain', connectgaps=False)])  # Disable connectgaps

# Adjust camera and layout
fig.update_layout(
    scene=dict(
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        zaxis_title='Elevation',
        aspectmode='manual',
        aspectratio=dict(x=1, y=3, z=0.1),
    )
)

# Save to HTML
html_string = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
with open('terrain_map_finland.html', 'w') as file:
    file.write(html_string)
