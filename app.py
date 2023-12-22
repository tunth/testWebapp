import plotly.graph_objects as go
import numpy as np
from numpy import log as ln

from dash import Dash, dcc, html, Input, Output, callback
import dash_core_components as dcc
import dash_html_components as html
# import dash_bootstrap_components as dbc

# app = dash.Dash(__name__)
app = Dash(__name__)
server = app.server



app.layout = html.Div(children=[
    html.H1(children='Welcome,'),

    # html.Div(children='''
    #     Dash: A web application framework for Python.
    # '''),
    html.Div([
        "Input: ",
        dcc.Input(id='load_input', value=1, type='text')
    ]),

    html.H1(children=''),

    dcc.Graph(
        id='example-graph'
        # figure=figure
    ),    
], style={'width': '55%', 'display': 'inline-block'}
)

@callback(
        Output('example-graph', 'figure'),
        [Input('load_input', 'value')]
)
def plotfunc(z):
    x = np.linspace(1, 10, 100)
    f = lambda x: 1/(ln(x)+0.1)
    g = lambda x: 0.5*x
    
    if z == '' : z=1
    else: z = float(z)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=x, y=f(x),
                name='Chi Phí')
    )
    figure.add_trace(
        go.Scatter(x=x, y=g(x),
                name='Lợi ích')
    )
    figure.add_traces(
        go.Scatter(x=[z], y=f([z]),
                   marker_size=20, marker_color="yellow"
                   )
    )
    figure.update_layout(
        title="Đồ thị",
        xaxis_title="Lợi ích",
        yaxis_title="Chi phí",
        xaxis=dict(range=[0,10]),
        yaxis=dict(range=[0,10]),
        paper_bgcolor="LightSteelBlue",
    )
    # figure.update_layout(width=int(width))
    # figure.update_layout(width=400)
    # print("Giá trị min cost:", min(f(x)))
    # figure.show()

    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
