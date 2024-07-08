import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from model import VelocityModel  # Ensure this model is properly implemented and efficient

app = dash.Dash(__name__)

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
    dcc.Interval(
    id='interval-component',
    interval=1*1000,  # in Millisekunden
    n_intervals=0,
    max_intervals=1  # Stellen Sie sicher, dass es nur einmal auslöst
),
    html.Div([
        html.Label("A/M-Verhältnis [m^2/kg]:", style={'display': 'block'}),
        dcc.Slider(
            id='param1-slider',
            min=0.01,
            max=2,
            step=0.01,  # Feinere Schrittbreite
            value=0.5,
            marks={
                0.5: '0.5',
                1.0: '1.0',
                1.5: '1.5',
                2.0: '2.0',
            }  # Benutzerdefinierte Beschriftungen an spezifischen Punkten
        ),
        dcc.Input(id='param1', type='number', value=0.5, style={'width': '100px', 'margin-left': '10px'}),
    ], style=styles['input-group']),

    html.Div([
        html.Label("Umgebungsdruck [Pa]:", style={'display': 'block'}),
        dcc.Slider(
            id='param2-slider',
            min=0,
            max=101325,
            step=1000,  # Feinere Schrittbreite
            value=101325,
            marks={
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
            value=2,
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

    html.Button('Predict and Plot', id='submit-button', n_clicks=0),    
    html.Div(id='output-container2', style={'padding': '20px', 'text-align': 'center'}),
    html.Div(id='output-container', style={'padding': '20px', 'text-align': 'center'}),
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
    [Output('output-container', 'children')] + [Output(f'plot{i+1}', 'figure') for i in range(4)],
    [Input('interval-component', 'n_intervals'), Input('submit-button', 'n_clicks')],
    [State('param1', 'value'),
     State('param2', 'value'),
     State('param3', 'value'),
     State('param4', 'value')]
)
def update_output_and_plots(n_intervals, n_clicks, param1, param2, param3, param4):
    trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'interval-component' and n_intervals == 1 or trigger == 'submit-button':
        try:
            params = [float(param1), float(param2), float(param3), float(param4)]
            model = VelocityModel(config="config_model.json", check_results=True)
            prediction = model.predict([params])

            # Namen der Parameter für die Tabelle
            param_names = ["A/M-Verhältnis [m^2/kg]", "Umgebungsdruck [Pa]", "Tankdurchmesser [m]", "Abstand zur Explosion [m]", "Prediction"]
            data = {'Parameter': param_names, 'Wert': [param1, param2, param3, param4, prediction[0]]}

            # Erstellen einer Tabelle zur Anzeige der Parameter und Vorhersage
            table = dash_table.DataTable(
                columns=[
                    {"name": "Parameter", "id": "Parameter"},
                    {"name": "Wert", "id": "Wert"},
                    #{"name": "Neuer Wert", "id": "new_value"}
                ],
                data=[{'Parameter': n, 'Wert': w} for n, w in zip(data['Parameter'], data['Wert'])], 
                style_table={'width': '50%', 'margin': 'auto', 'margin-top': '20px'}
            )

            # Update plots
            figures = []
            for i, val in enumerate(params):
                if i == 0:
                    x = np.linspace(0.01, 2, 100)
                elif i == 1:
                    x = np.linspace(0, 101325, 100)
                elif i == 2:
                    x = np.linspace(0.1, 8, 100)
                else:
                    x = np.linspace(0, 8, 100)
                y = [model.predict([[x[j] if i==k else params[k] for k in range(4)]])[0] for j in range(100)]
                fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
                fig.update_layout(
                    title=f'Vorhersage mit variierendem {param_names[i]}',
                    xaxis_title=param_names[i],
                    yaxis_title='Fragmentgeschwindigkeit [m/s]'
                )
                figures.append(fig)

            return [table] + figures
        except Exception as e:
            return [html.H2(f"An error occurred: {str(e)}")] + [go.Figure() for _ in range(4)]
    return [html.Div()] + [go.Figure() for _ in range(4)]


def update_input_values(s1, s2, s3, s4):
    return s1, s2, s3, s4

if __name__ == '__main__':
    app.run_server(debug=True)
