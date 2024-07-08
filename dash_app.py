import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from model import VelocityModel  # Ensure this model is properly implemented and efficient

app = dash.Dash(__name__)

line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']


# Define the CSS styles for a cleaner look
styles = {
    'container': {
        'padding': '10px',
        'margin': '10px'
    },
    'input-group': {
        'margin': '20px 0'
    }
}

app.layout = html.Div([
    html.H2("Modell zur Bestimmung der aufgezwungenen Fragmentgeschwindigkeit infolge eines explosiven Zerfalls eines Trägersystems ", style={'text-align': 'center'}),
    html.Hr(),
    dcc.Store(id='plot-data-store', storage_type='memory'),  # Speichert die Plotdaten
    dcc.Store(id='table-data-store', storage_type='memory'),  # Speichert die Tabellendaten
    dcc.Interval(
    id='interval-component',
    interval=1*1000,  # in Millisekunden
    n_intervals=0,
    max_intervals=1  # Stellen Sie sicher, dass es nur einmal auslöst
),
    html.Div([
        html.Label("A/M-Verhältnis [m²/kg]:", style={'display': 'block'}),
        dcc.Slider(
            id='param1-slider',
            min=0.01,
            max=2,
            step=0.01,  # Feinere Schrittbreite
            value=0.1,
            marks={
                0.01: '0.01',
                0.5: '0.5',
                1.0: '1.0',
                1.5: '1.5',
                2.0: '2.0',
            }  # Benutzerdefinierte Beschriftungen an spezifischen Punkten
        ),
        dcc.Input(id='param1', type='number', value=0.1, style={'width': '100px', 'margin-left': '10px'}),
    ], style=styles['input-group']),

    html.Div([
        html.Label("Umgebungsdruck [Pa]:", style={'display': 'block'}),
        dcc.Slider(
            id='param2-slider',
            min=0,
            max=101325,
            step=100,  # Feinere Schrittbreite
            value=101325,
            marks={
                0: '0',
                25000: '25k',
                50000: '50k',
                75000: '75k',
                100000: '100k',
            }  # Benutzerdefinierte Beschriftungen an spezifischen Punkten
        ),
        dcc.Input(id='param2', type='number', value=101325, style={'width': '100px', 'margin-left': '10px'}),
    ], style=styles['input-group']),

    html.Div([
        html.Label("Tankdurchmesser [m]:", style={'display': 'block'}),
        dcc.Slider(
            id='param3-slider',
            min=0.1,
            max=8,
            step=0.01,  # Feinere Schrittbreite
            value=4,
            marks={
                0.1: '0.1',
                2: '2',
                4: '4',
                6: '6',
            }  # Benutzerdefinierte Beschriftungen an spezifischen Punkten
        ),
        dcc.Input(id='param3', type='number', value=2, style={'width': '100px', 'margin-left': '10px'}),
    ], style=styles['input-group']),

    html.Div([
        html.Label("Abstand zur Explosion [m]:", style={'display': 'block'}),
        dcc.Slider(
            id='param4-slider',
            min=0,
            max=25,
            step=0.01,  # Feinere Schrittbreite
            value=0,
            marks={
                0: '0',
                5: '5',
                10: '10',
                15: '15',
                20: '20',
                25: '25'
            }  # Benutzerdefinierte Beschriftungen an spezifischen Punkten
        ),
        dcc.Input(id='param4', type='number', value=0.5, style={'width': '100px', 'margin-left': '10px'}),
    ], style=styles['input-group']),  
    html.Div(id='output-container2', style={'padding': '20px', 'text-align': 'center'}),
    html.Button('Plot', id='submit-button', n_clicks=0),
    html.Div(id='output-container', style={'padding': '20px', 'text-align': 'center'}),
    dash_table.DataTable(id='parameter-table'),
    html.Div([
        dcc.Graph(id='plot1'),
        dcc.Graph(id='plot2'),
        dcc.Graph(id='plot3'),
        dcc.Graph(id='plot4')
    ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}),
    
], style={'margin': 'auto', 'width': '80%', 'padding': '20px', 'box-shadow': '0px 0px 20px #AAA', 'font-family': 'Arial, sans-serif'})


@app.callback(
    [
        Output('param1', 'value'),
        Output('param2', 'value'),
        Output('param3', 'value'),
        Output('param4', 'value'),
        Output('output-container2', 'children')  # Output for displaying predictions
    ],
    [
        Input('param1-slider', 'value'),
        Input('param2-slider', 'value'),
        Input('param3-slider', 'value'),
        Input('param4-slider', 'value')
    ]
)
def update_input_values_and_predict(param1_slider, param2_slider, param3_slider, param4_slider):
    params = [param1_slider, param2_slider, param3_slider, param4_slider]
    try:
        model = VelocityModel(config="config_model.json", check_results=True)
        prediction = model.predict([params])
        prediction_text = [html.H2(f"Vorhergesagte Geschwindigkeit: {prediction[0]} m/s"), html.Hr(),html.Hr()]
    except Exception as e:
        prediction_text = html.H2(f"An error occurred while predicting: {str(e)}")

    return param1_slider, param2_slider, param3_slider, param4_slider, prediction_text

@app.callback(
       [ Output('plot-data-store', 'data'),
        Output('table-data-store', 'data'),
        Output('parameter-table', 'columns'),
        Output('parameter-table', 'data')
    ] + [Output(f'plot{i+1}', 'figure') for i in range(4)],
    [Input('interval-component', 'n_intervals'), Input('submit-button', 'n_clicks')],
    [State('param1', 'value'),
     State('param2', 'value'),
     State('param3', 'value'),
     State('param4', 'value'),
     State('plot-data-store', 'data'),
     State('table-data-store', 'data')]
)
def update_plots_and_table(x,n_clicks, param1, param2, param3, param4, plot_data, table_data):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    current_style_index = n_clicks % len(line_styles) 
    params = [param1, param2, param3, param4]
    model = VelocityModel(config="config_model.json")
    prediction = model.predict([params])

    # Update plot data
    if plot_data is None:
        plot_data = {f'plot{i+1}': [] for i in range(4)}

    for i in range(4):
        #x_vals
        if i == 0:
            x_vals = np.linspace(0.01, 2, 100)
        elif i == 1:
            x_vals = np.linspace(0, 101325, 100)
        elif i == 2:
            x_vals = np.linspace(2, 6, 100)
        else:
            x_vals = np.linspace(0, 8, 100)
        y_vals = [model.predict([[x if j == i else params[j] for j in range(4)]])[0] for x in x_vals]
        plot_data[f'plot{i+1}'].append({
            'x': x_vals, 'y': y_vals, 'type': 'scatter', 'mode': 'lines',
            'name': f'Fragment {n_clicks}',
            'line': {'dash': line_styles[current_style_index]}
        })

    # Update table data
    if table_data is None:
        table_data = {'columns': [{'name': 'Parameter', 'id': 'Parameter'}], 'data': []}
    table_data['columns'].append({'name': f'Fragment {n_clicks}', 'id': f'Fragment {n_clicks}'})
    if not table_data['data']:
        table_data['data'] = [{'Parameter': 'A/M-Verhältnis [m²/kg]'}, {'Parameter': 'Umgebungsdruck [Pa]'}, {'Parameter': 'Tankdurchmesser [m]'}, {'Parameter': 'Abstand zur Explosion [m]'}, {'Parameter': 'Fragmentgeschwindigkeit [m/s]'}]
    for i, p in enumerate(params + [prediction[0]]):
        table_data['data'][i][f'Fragment {n_clicks}'] = p

    figures = []
    for i in range(4):
        fig = go.Figure(data=[
            go.Scatter(x=data['x'], y=data['y'], mode='lines', name=data['name'],
                       line={'dash': data['line']['dash']})
            for data in plot_data[f'plot{i+1}']
        ])
        fig.update_layout(
            title=f'Vorhersage mit variierendem {table_data["data"][i]["Parameter"]}',
            xaxis_title=table_data["data"][i]["Parameter"],
            yaxis_title='Fragmentgeschwindigkeit [m/s]'
        )
        figures.append(fig)

    return [plot_data, table_data, table_data['columns'], table_data['data']] + figures


def update_input_values(s1, s2, s3, s4):
    return s1, s2, s3, s4

if __name__ == '__main__':
    app.run_server(debug=True)
