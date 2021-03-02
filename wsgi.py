import numpy as np
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import json
import matplotlib

from data_handler import stability_data_handler, FQ_data_handler, TUNE_X_data_handler, TUNE_Y_data_handler


##### CONSTANTS ################################################################
COLORS = ["red", "blue", "green", "orange", "cyan"]
FIGURE_HEIGHT = 1000
FIGURE_WIDTH = 1200
################################################################################


##### DASH Framework ###########################################################
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
application = app.server
################################################################################

###### PLOTS LAYOUT ############################################################
blocks = [
    dbc.Col([
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id="fig_action",
                    figure=go.Figure()
                )
            ]),
            dbc.Col([
                dcc.Graph(
                    id="fig_frequency",
                    figure=go.Figure()
                )
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label(
                    children="Epsilon"
                ),
                dcc.Dropdown(
                    id="drop_epsilon",
                    options=[{'label': str(s), 'value': s} for s in  stability_data_handler.get_param_options(
                        "epsilon")],
                    value=stability_data_handler.get_param_options("epsilon")[0],
                    multi=False,
                    clearable=False
                ),
                dbc.Label(
                    children="Mu"
                ),
                dcc.Dropdown(
                    id="drop_mu",
                    options=[{'label': str(s), 'value': s} for s in stability_data_handler.get_param_options(
                        "mu")],
                    value=stability_data_handler.get_param_options("mu")[0],
                    multi=False,
                    clearable=False
                ),
                dbc.Label(
                    children="N turns"
                ),
                dcc.Dropdown(
                    id="drop_nturns",
                    options=[{'label': str(s), 'value': s} for s in FQ_data_handler.get_param_options(
                        "turns")],
                    value=FQ_data_handler.get_param_options("turns")[0],
                    multi=False,
                    clearable=False
                )
            ]),
            dbc.Col(dbc.FormGroup([
                dbc.Row([
                    dbc.Label(
                        children="Tolerance"
                    ),
                    dcc.Input(
                        id="input_tolerance",
                        type="number",
                        value=1e-4 
                    )
                ]),
                dbc.Row([
                    dbc.Label(
                        children="Min resonance"
                    ),
                    dcc.Input(
                        id="input_minres",
                        type="number",
                        value=3
                    ),
                    dbc.Label(
                        children="Max resonance"
                    ),
                    dcc.Input(
                        id="input_maxres",
                        type="number",
                        value=6
                    )
                ]),
                dbc.Row([
                    dbc.Label(
                        children="Known X tune"
                    ),
                    dcc.Input(
                        id="xtune",
                        type="number",
                        value=0.168
                    ),
                    dbc.Label(
                        children="Known Y tune"
                    ),
                    dcc.Input(
                        id="ytune",
                        type="number",
                        value=0.201
                    )
                ])
            ])),
        ])
    ])
]

################################################################################

##### FINAL LAYOUT #############################################################
app.layout = html.Div([
    dbc.Toast(
        [html.P("Plot(s) updated!", className="mb-0")],
        id="notification-toast",
        header="Notification",
        icon="primary",
        dismissable=True,
        is_open=False,
        duration=4000,
        style={"position": "fixed-top", "top": 66, "right": 10, "width": 350},
    ),
    blocks[0]
])
################################################################################


##### CALLBACKS ################################################################

#### Grab data and create figure ####

@app.callback(
    Output('fig_action', 'figure'),
    [
        Input('drop_epsilon', 'value'),         # 0
        Input('drop_mu', 'value'),              # 1
        Input('drop_nturns', 'value'),          # 2
        Input('input_tolerance', 'value'),      # 3
        Input('input_minres', 'value'),         # 4
        Input('input_maxres', 'value'),         # 5
        Input('xtune', 'value'),                # 6
        Input('ytune', 'value')                 # 7
    ]
)
def update_action_plot(*args):
    parameters = {
        "mu": args[1],
        "epsilon": args[0],
        "turns": args[2],
    }
    data_x = TUNE_X_data_handler.get_data(parameters)
    data_y = TUNE_Y_data_handler.get_data(parameters)
    data = np.empty((500,500)) * np.nan
    extra_data_x = np.zeros((500,500))
    extra_data_y = np.zeros((500,500))

    for i in list(range(args[4], args[5] + 1)):
        for j in range(0, i+1):
            nx = j
            ny = i - j

            bool_mask = np.modf(np.absolute(+ nx * data_x + ny * data_y))[0] < args[3]
            data[bool_mask] = i
            extra_data_x[bool_mask] = nx
            extra_data_y[bool_mask] = ny

            bool_mask = np.modf(np.absolute(+ nx * data_x - ny * data_y))[0] < args[3]
            data[bool_mask] = i
            extra_data_x[bool_mask] = nx
            extra_data_y[bool_mask] = -ny

    actual_max_res = int(np.nanmax(data))
    actual_min_res = int(np.nanmin(data))
    n_resonances = (actual_max_res - actual_min_res) + 1
    interval = 1 / (n_resonances)
    colorscale = []
    for i, j in enumerate(range(actual_min_res, actual_max_res + 1)):
        colorscale.append([interval * i, COLORS[j % len(COLORS)]])
        colorscale.append([interval * (i + 1), COLORS[(j) % len(COLORS)]])

    fig = go.Figure({
        'data': [{
            'z': data,
            'x': np.linspace(0, 1, 500),
            'y': np.linspace(0, 1, 500),
            'hoverongaps': False,
            'type': 'heatmap',
            'customdata': np.dstack((
                [extra_data_x, extra_data_y]
            )),
            'hovertemplate': "<br>".join([
                "Resonance Order: %{z}",
                "n_x: %{customdata[0]}",
                "n_y: %{customdata[1]}"
            ]),
            'colorscale': colorscale,
            'colorbar':dict(
                dtick=1
            )
        }]
    })

    fig.update_layout(
        title="Resonance plot - Action space - resonance order in colorbar",
        xaxis_title="X_0",
        yaxis_title="Y_0"
    )
    fig.update_layout(width=int(FIGURE_WIDTH))
    fig.update_layout(height=int(FIGURE_HEIGHT))
    return fig


def filter_01(x, y):
    mask = np.logical_and(
        np.logical_and(x >= -0.01, x <= 1.01),
        np.logical_and(y >= -0.01, y <= 1.01)
    )
    return x[mask], y[mask]

@app.callback(
    Output('fig_frequency', 'figure'),
    [
        Input('drop_epsilon', 'value'),         # 0
        Input('drop_mu', 'value'),              # 1
        Input('drop_nturns', 'value'),          # 2
        Input('input_tolerance', 'value'),      # 3
        Input('input_minres', 'value'),         # 4
        Input('input_maxres', 'value'),         # 5
        Input('xtune', 'value'),                # 6
        Input('ytune', 'value')                 # 7
    ]
)
def update_frequency_plot(*args):
    parameters = {
        "mu": args[1],
        "epsilon": args[0],
        "turns": args[2],
    }
    data_x = TUNE_X_data_handler.get_data(parameters)
    data_y = TUNE_Y_data_handler.get_data(parameters)

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=data_x.flatten(),
            y=data_y.flatten(),
            name="data",
            mode="markers",
            marker_color="black"
        )
    )
    x = np.linspace(0, 1, 1000)
    for i in list(range(args[4], args[5] + 1)):
        for j in range(1, i):
            nx = j
            ny = i - j
            for q in range(0, i+1):
                newx, y = filter_01(x, q/ny - nx / ny * x)
                fig.add_trace(go.Scattergl(
                    x=newx,
                    y=y,
                    mode='lines',
                    marker_color=COLORS[i % len(COLORS)],
                    name="Resonance {}".format(i),
                    showlegend=(True if q == 0 and j == 1 else False),
                    hoverinfo="skip"
                ))
                newx, y = filter_01(x, q/(-ny) - nx / (-ny) * x)
                fig.add_trace(go.Scattergl(
                    x=newx,
                    y=y,
                    mode='lines',
                    marker_color=COLORS[i % len(COLORS)],
                    name="Resonance {} - {} - bis".format(i, q),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                newx, y = filter_01(x, q/ny - (-nx) / ny * x)
                fig.add_trace(go.Scattergl(
                    x=newx,
                    y=y,
                    mode='lines',
                    marker_color=COLORS[i % len(COLORS)],
                    name="Resonance {} - {} - ter".format(i, q),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                newx, y = filter_01(x, q/(-ny) - (-nx) / (-ny) * x)
                fig.add_trace(go.Scattergl(
                    x=newx,
                    y=y,
                    mode='lines',
                    marker_color=COLORS[i % len(COLORS)],
                    name="Resonance {} - {} - quater".format(i, q),
                    showlegend=False,
                    hoverinfo="skip"
                ))
        for j in range(i+1):
            fig.add_vline(j/i, line_color=COLORS[i % len(COLORS)])
            fig.add_hline(j/i, line_color=COLORS[i % len(COLORS)])

    fig.add_trace(go.Scattergl(
        x=x,
        y=np.ones_like(x) * args[7],
        name="Working Frequency",
        mode='lines',
        line_color="grey",
        showlegend=True,
        hoverinfo="skip"
    ))
    fig.add_vline(args[6], line_color="grey")

    fig.update_xaxes(range=[0.0, 1.0])
    fig.update_yaxes(range=[0.0, 1.0])
    fig.update_layout(width=int(FIGURE_WIDTH))
    fig.update_layout(height=int(FIGURE_HEIGHT))

    fig.update_layout(
        title="Resonance plot - Frequency space",
        xaxis_title="X tune [2pi units]",
        yaxis_title="Y tune [2pi units]"
    )

    return fig


@app.callback(
    Output("notification-toast", "is_open"),
    [
        Input("fig_action", 'figure'),
        Input("fig_frequency", 'figure'),
    ]
)
def update_toast(*p):
    return True

################################################################################


##### RUN THE SERVER ###########################################################
if __name__ == '__main__':
    app.run_server(port=8080, debug=True)
################################################################################
