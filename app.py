from dash import Dash, dash_table, dcc, html, Input, Output, State, callback
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from numpy import log as ln
import datetime as dt

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# data input
data_input = {
    'DMA_code': ['NB04031', 'NB04021'],
    'DMA_name': ['189C Tôn Thất Thuyết', 'Hẻm 88 Nguyễn Khoái'],
    'Lav': [19.3178191489362, 65.033242150333],
    'Cmax': [5.51, 5.43]
}
df_input = pd.DataFrame(data_input)

class grapp_ell:
    def __init__(self, dma):
        self.Cmax = df_input[df_input['DMA_code']==dma]['Cmax'].iloc[0]
        self.Lav = df_input[df_input['DMA_code']==dma]['Lav'].iloc[0]
        self.Ca = 42905.1757260401
        self.Lb = 5.41
        self.Lp = 100.9
        self.unit_price = 10000
        self.min_cost = 10482.68
        self.L = np.linspace(self.Cmax, 100, 50)
        self.K = self.Ca/(ln((self.Lav-self.Lb)/(self.Lp-self.Lb)))
    def C(self, L):
      return ln((L-self.Lb)/(self.Lp-self.Lb))
    def ALC(self, L):
      return [self.K]*self.C(L) + self.min_cost if isinstance(L, (np.ndarray, np.generic) ) else self.K*self.C(L) + self.min_cost 
    def MCW(self, L):
      return L*self.unit_price
    def ELL(self, L):
      (self.ALC(L) + self.MCW(L))

p1 = grapp_ell('NB04031')
# L = np.linspace(p1.Cmax, 100, 50)
arr = np.array([list(p1.L), list(p1.ALC(p1.L)), list(p1.MCW(p1.L)), list(p1.ALC(p1.L) + p1.MCW(p1.L))])
arr = np.transpose(arr)
df_ell = pd.DataFrame(arr, columns=['L rate', 'ALC', 'MCW', 'ELL'])
# df_ell.head()

currentdatetime = dt.datetime.now()
data = {'Datetime': [],
        'Lav': [],
        'ALC': [],
        'MCW': [],
        'ELL': []}
df = pd.DataFrame(data)

app.layout = dbc.Container(children=[
    html.H1('Welcome!'),
    html.Div([
        dcc.Dropdown(df_input.DMA_code.unique(), id='pandas-dropdown-2'),
        html.Br(),
        html.Div(id='pandas-output-container-2'),
        html.Br(),
    ]),
    html.Div(
       dbc.Row(
            [
                dbc.Col(
                   html.Div(
                    dcc.Input(
                            id='input1',
                            placeholder='Input the Lav to calculate related indicators..',
                            type='number',
                            value='',
                            style={'padding': 10}
                        )
                    ),md=3,
                ),
                dbc.Col(html.Div(
                   html.Button('Add Row', id='editing-rows-button', n_clicks=0)
                    ), md=5,
                ),
                dbc.Col(html.Div(
                    html.Button('Export to Excel', id='save_to_csv', n_clicks=0),  
                   ), md=4
                ),
                dbc.Col(
                    # Create notification when saving to excel
                    html.Div(id='placeholder', children=[]),
                ),
                dbc.Col(
                    dcc.Store(id="store", data=0),
                ),
                dbc.Col(
                    dcc.Interval(id='interval', interval=1000),
                )
            ]
        ),
    ) ,  
    html.H1(''),

    dbc.Row(
       [
        dbc.Col(
            dcc.Graph(
                id='ell-graph'
                # figure=figure
            ), md=8
        # , style={'width': '89%', 'display': 'inline-block'}),  
        ), 
        dbc.Col(
            dash_table.DataTable(
                id='adding-rows-table',
                columns=[{'name': i, 'id': i} for i in df.columns],
                data=df.to_dict('records'),
                editable=True,
                row_deletable=True
            ), md=4,
        # , style={'width': '11%', 'display': 'inline-block'}
        )
    ]),

    html.H1(''),

    dcc.Graph(id='adding-rows-graph')
], style={'width': '75%', 'display': 'inline-block'}
)

@callback(
    Output('pandas-output-container-2', 'children'),
    Input('pandas-dropdown-2', 'value')
)
def update_output(value):
    a=''
    if value is not None:
        a = df_input[df_input['DMA_code']==value]['Lav'].iloc[0]
        a = f' has Lav {a}'
    return f'You have selected {value}{a}'

@callback(
    Output('adding-rows-table', 'data'),
    Input('editing-rows-button', 'n_clicks'),
    Input('pandas-dropdown-2', 'value'),
    State('adding-rows-table', 'data'),
    State('adding-rows-table', 'columns'),
    State('input1', 'value')
    # State('input2', 'value')
    )
def add_row(n_clicks, dma, rows, columns, value1):
    if n_clicks > 0 and value1 != '' and value1 is not None and dma is not None:
        # rows.append({c['id']: '' for c in columns})
        p2 = grapp_ell(dma)
        rows.append({'Datetime': currentdatetime.strftime('%Y-%m-%d %H%M'), 'Lav': value1, 'ALC': p2.ALC(value1), 'MCW': p2.MCW(value1), 'ELL': p2.ALC(value1) + p2.MCW(value1)})
    return rows

@callback(
        Output('ell-graph', 'figure'),
        # Output('outputResult', 'val'),
        Input('input1', 'value'),
        Input('pandas-dropdown-2', 'value')
)
def plotfunc(z, dma):
    figure = go.Figure()

    if dma is not None:
        p2 = grapp_ell(dma)
        Lav = p2.Lav
        Cmax = p1.Cmax
        
        Ca = p1.Ca
        Lb = p1.Lb
        Lp = p1.Lp
        unit_price = p1.unit_price
        min_cost = p1.min_cost

        L = p2.L
        
        if z == '' or z is None: z=10

        figure.add_trace(
            go.Scatter(x=p2.L, y=p1.ALC(p2.L),
                    name='Chi Phí ALC')
        )
        figure.add_trace(
            go.Scatter(x=p2.L, y=p2.ALC(p2.L) + p2.MCW(p2.L),
                    name='Chi Phi ELL')
        )
        figure.add_traces(
            go.Scatter(x=[z], y=[p2.ELL(z)],
                    marker_size=20, marker_color="yellow"
                    )
        )
        figure.update_layout(
            title="Đồ thị",
            xaxis_title="Lợi ích",
            yaxis_title="Chi phí",
            xaxis=dict(range=[0,100]),
            yaxis=dict(range=[0,1100000]),
            paper_bgcolor="LightSteelBlue",
        )
    return figure

@app.callback(
    [Output('placeholder', 'children'),
     Output("store", "data")],
    [Input('save_to_csv', 'n_clicks'),
     Input("interval", "n_intervals")],
    [State('adding-rows-table', 'data'),
     State('store', 'data')]
)
def df_to_csv(n_clicks, n_intervals, dataset, s):
    output = html.Plaintext("The data has been saved to your folder.",
                            style={'color': 'green', 'font-weight': 'bold', 'font-size': 'large'})
    no_output = html.Plaintext("", style={'margin': "0px"})

    input_triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if input_triggered == "save_to_csv":
        s = 6
        df = pd.DataFrame(dataset)
        df.to_csv("Data_test.csv", index=False, mode='a', header=False)
        return output, s
    elif input_triggered == 'interval' and s > 0:
        s = s-1
        if s > 0:
            return output, s
        else:
            return no_output, s
    elif s == 0:
        return no_output, s

@callback(
    Output('adding-rows-graph', 'figure'),
    Input('adding-rows-table', 'data'),
    Input('adding-rows-table', 'columns'))
def display_output(rows, columns):
    return {
        'data': [{
            'type': 'heatmap',
            'z': [[row.get(c['id'], None) for c in columns] for row in rows],
            'x': [c['name'] for c in columns]
        }]
    }


if __name__ == '__main__':
    app.run(debug=True)
