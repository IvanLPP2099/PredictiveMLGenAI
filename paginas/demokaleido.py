import plotly.express as px

fig = px.line(x=[1, 2, 3], y=[4, 5, 6])
fig.write_image("test_fig.png", width=900, height=500)
print("✅ Imagen guardada correctamente")
