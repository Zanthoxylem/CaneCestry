import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import flask
from flask import send_file
import tempfile
import numpy as np
import graphviz
from flask import Flask
import uuid
from collections import defaultdict
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# ---- NEW: Numba imports
import numba
from numba import njit

# A threshold to skip clustering if matrix exceeds this size
MAX_CLUSTER_SIZE = 300

# Initialize Flask server
server = Flask(__name__)

# =============================================================================
# 1. Store Built-In (Default) Sugarcane Data and Create a Copy for `df`
# =============================================================================
default_df = pd.read_csv('/home/Zanthoxylum2117/Canecestry/Pedigree_24.txt', sep="\t")
df = default_df.copy()

# Create filtered_df from df
parents_set = set(df['MaleParent'].tolist() + df['FemaleParent'].tolist())
mask = (
    ~df['LineName'].isin(parents_set)
    & (df['MaleParent'].isna() | df['MaleParent'].isnull())
    & (df['FemaleParent'].isna() | df['FemaleParent'].isnull())
)
filtered_df = df[~mask]

# Global variables
current_amatrix = None
in_memory_matrices = {}
temp_progeny_count = 0
temp_progeny_list = []
dummy_progeny_list = {}
full_tree_data = {}
full_tree_kinship_values = {}

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SANDSTONE, "https://use.fontawesome.com/releases/v5.10.2/css/all.css"],
    server=server,
    suppress_callback_exceptions=True
)

# Custom CSS styles
CUSTOM_CSS = {
    'container': {'padding': '20px', 'margin-top': '20px', 'background-color': '#f9f9f9'},
    'header': {
        'textAlign': 'center',
        'padding': '10px',
        'color': '#2c3e50',
        'font-family': 'Arial',
        'font-weight': 'bold',
        'font-size': '40px'
    },
    'button': {'margin': '10px', 'font-weight': 'bold', 'background-color': '#2980b9', 'color': 'white'},
    'table': {'overflowX': 'auto', 'margin-bottom': '20px', 'border': '1px solid #ccc'},
    'image': {'width': '100%', 'padding': '10px'},
    'dropdown': {'font-weight': 'bold', 'color': '#2980b9'},
    'static-button': {
        'position': 'absolute',
        'top': '10px',
        'right': '10px',
        'border-radius': '50%',
        'width': '75px',
        'height': '75px',
        'font-size': '40px',
        'textAlign': 'center',
        'lineHeight': '50px',
        'background-color': '#2980b9',
        'color': 'white',
        'border': 'none'
    },
    'home-button': {
        'position': 'absolute',
        'left': '10px',
        'top': '10px',
        'border-radius': '50%',
        'width': '75px',
        'height': '75px',
        'font-size': '40px',
        'textAlign': 'center',
        'lineHeight': '50px',
        'background-color': '#2980b9',
        'color': 'white',
        'border': 'none'
    }
}


def base_layout(content):
    """Generates the base layout with common structure."""
    return dbc.Container([
        dbc.Row([
            # Home button
            dbc.Col(
                html.Button(
                    html.I(className="fas fa-home"),
                    id="home-button",
                    style=CUSTOM_CSS['home-button']
                )
            ),
            # Clear Data (trash) button
            dbc.Col(
                html.Button(
                    html.I(className="fas fa-trash"),
                    id="clear-data-button",
                    style={
                        **CUSTOM_CSS['static-button'],
                        'right': '100px',  # Shift left from the "?" button
                        'background-color': '#e74c3c'
                    }
                ),
                width="auto"
            ),
        ]),
        dbc.Row(
            dbc.Col(
                html.H1("CaneCestry", className="app-header",
                        style={**CUSTOM_CSS['header'], 'font-size': '60px'})  # Bigger font size
            )
        ),
        dbc.Row(
            dbc.Col(html.Div(id='common-content', children=content))
        ),
        dbc.Row(
            dbc.Col(html.Div(id='page-specific-content'))
        ),
        html.Button("?", id="open-info-modal", style=CUSTOM_CSS['static-button']),
        dbc.Modal([
            dbc.ModalHeader("Information"),
            dbc.ModalBody(
                html.Div([
                    html.H4("Overview"),
                    html.P("CaneCestry 2.0 is a tool for managing and analyzing pedigree data in sugarcane breeding programs. It offers functionalities to visualize family trees, compute kinship matrices, and explore progeny."),
                    html.H4("Main Page"),
                    html.Ul([
                        html.Li("Select Line Names: Choose one or more line names to analyze."),
                        html.Li("Get Matrix: Generate the kinship matrix for selected lines."),
                        html.Li("Download Full Matrix: Download the full kinship matrix."),
                        html.Li("Download Subset Matrix: Download a subset of the kinship matrix."),
                        html.Li("Heatmap: Visualize the kinship matrix as a heatmap.")
                    ]),
                    html.H4("Progeny Finder"),
                    html.Ul([
                        html.Li("Lookup Progeny of Specific Pairing: Find progeny resulting from a specific male and female parent."),
                        html.Li("Lookup Progeny of Single Parent: Find all progeny of a single parent."),
                        html.Li("Generate Family Tree: Visualize the ancestry of a selected line."),
                        html.Li("Generate Descendant Tree: Visualize the descendants of a selected line."),
                        html.Li("Generate Combined Family Tree for Two Lines: Visualize the combined ancestry of two selected lines.")
                    ])
                ])
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-info-modal", className="ml-auto")
            )
        ], id="info-modal", size="lg", is_open=False)
    ], fluid=True, style=CUSTOM_CSS['container'])


def splash_page_layout():
    """Layout for the splash page with a modal overlay to choose data source."""
    data_modal = dbc.Modal(
        [
            dbc.ModalHeader("Choose Your Data Source"),
            dbc.ModalBody([
                dcc.RadioItems(
                    id='splash-data-choice',
                    options=[
                        {'label': 'Use Built-In Sugarcane Data', 'value': 'example'},
                        {'label': 'Upload Your Own Data', 'value': 'upload'},
                    ],
                    value='example',
                    labelStyle={'display': 'block', 'marginBottom': '10px'},
                    style={'marginBottom': '20px'}
                ),
                html.Div(
                    [
                        html.P(
                            "Upload a .csv (or .txt with tab separator) with columns: LineName, MaleParent, FemaleParent",
                            style={'fontWeight': 'bold'}
                        ),
                        dcc.Upload(
                            id='splash-upload-data',
                            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False
                        ),
                        html.Div(id='splash-upload-status', style={'marginTop': '10px', 'color': 'green'}),
                    ],
                    id='splash-upload-section',
                    style={'display': 'none'}
                ),
            ]),
            dbc.ModalFooter(
                dbc.Button("Proceed", id="splash-modal-proceed-btn", color="primary")
            ),
        ],
        id='splash-data-modal',
        is_open=True,
        backdrop='static',
        keyboard=False
    )

    content = [
        data_modal,
        dbc.Row(
            dbc.Col(
                html.P(
                    "Please select one of the following options to proceed:",
                    style={'textAlign': 'center', 'fontSize': '24px', 'marginBottom': '20px'}
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                html.A(
                    "Generate Kinship Matrix",
                    href="/main-page",
                    className="btn",
                    style={"width": "300px", "height": "80px", "fontSize": "20px",
                           "marginBottom": "10px",
                           "backgroundColor": "#2980b9", "color": "white",
                           "borderRadius": "10px", "fontWeight": "bold",
                           "display": "flex", "justifyContent": "center",
                           "alignItems": "center", "textDecoration": "none"}
                ),
                width="auto"
            ),
            justify="center"
        ),
        dbc.Row(
            dbc.Col(
                html.A(
                    "Pedigree Explorer",
                    href="/progeny-finder",
                    className="btn",
                    style={"width": "300px", "height": "80px", "fontSize": "20px",
                           "marginBottom": "10px",
                           "backgroundColor": "#2980b9", "color": "white",
                           "borderRadius": "10px", "fontWeight": "bold",
                           "display": "flex", "justifyContent": "center",
                           "alignItems": "center", "textDecoration": "none"}
                ),
                width="auto"
            ),
            justify="center"
        ),
        dbc.Row(
            dbc.Col(
                html.A(
                    "Upload Pedigree Data",
                    href="/upload",
                    className="btn",
                    style={"width": "300px", "height": "80px", "fontSize": "20px",
                           "marginBottom": "10px",
                           "backgroundColor": "#2980b9", "color": "white",
                           "borderRadius": "10px", "fontWeight": "bold",
                           "display": "flex", "justifyContent": "center",
                           "alignItems": "center", "textDecoration": "none"}
                ),
                width="auto"
            ),
            justify="center"
        ),
        dbc.Row(
            dbc.Col(
                html.A(
                    "Add Pedigree Entries",
                    href="/dummy-progeny-matrix",
                    className="btn",
                    style={"width": "300px", "height": "80px", "fontSize": "20px",
                           "marginBottom": "10px",
                           "backgroundColor": "#2980b9", "color": "white",
                           "borderRadius": "10px", "fontWeight": "bold",
                           "display": "flex", "justifyContent": "center",
                           "alignItems": "center", "textDecoration": "none"}
                ),
                width="auto"
            ),
            justify="center"
        )
    ]
    return base_layout(content)


def upload_page_layout():
    """Layout for the upload page."""
    example_df = df.head()
    example_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in example_df.columns],
        data=example_df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'padding': '10px'}
    )

    content = [
        dbc.Row(dbc.Col(html.H1("Upload Your Pedigree Data", className="app-header", style=CUSTOM_CSS['header']))),
        dbc.Row(dbc.Col(
            html.P("Please upload your pedigree data file. The accepted format is .CSV. "
                   "The file should contain the following columns:", style={'marginTop': '10px'})
        )),
        dbc.Row(dbc.Col(html.Ul([
            html.Li("LineName: The name of the line."),
            html.Li("MaleParent: The name of the male parent."),
            html.Li("FemaleParent: The name of the female parent.")
        ]))),
        dbc.Row(dbc.Col(html.P("Below is an example of the correct format:", style={'marginTop': '10px'}))),
        dbc.Row(dbc.Col(example_table, width=12)),
        dbc.Row(dbc.Col(dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ))),
        dbc.Row(dbc.Col(html.Div(id='upload-status', style={'textAlign': 'center', 'marginTop': '20px'}))),
        dbc.Row(dbc.Col(html.A("Proceed to Main Page", href="/main-page",
                               className="btn btn-primary", style=CUSTOM_CSS['button']))),
    ]
    return base_layout(content)


def main_page_layout():
    """Improved layout for the main page."""
    content = [
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Generate Kinship Matrix",
                    className="app-header",
                    style=CUSTOM_CSS['header']
                )
            )
        ),
        dbc.Row([
            dbc.Col(html.Label(
                "Select or Paste Line Names (comma-separated):",
                style={'font-weight': 'bold', 'margin-bottom': '10px'}
            ), width=12),
            dbc.Col(
                dcc.Dropdown(
                    id='line-name-dropdown',
                    options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                    multi=True,
                    placeholder="Type to search line names...",
                    style=CUSTOM_CSS['dropdown']
                ),
                width=12
            ),
            dbc.Col(
                dcc.Input(
                    id='paste-line-names',
                    type='text',
                    placeholder="Paste line names here...",
                    style={'margin-top': '10px', 'width': '100%'}
                ),
                width=12
            )
        ], className="mb-4"),
        dbc.Row(
            dbc.Col(
                html.Ul(id="selected-line-names-list", children=[]),
                style={'margin-top': '10px'}
            )
        ),
        dbc.Row([
            dbc.Col(
                html.Button(
                    "Generate Kinship Matrix",
                    id="generate-amatrix-button",
                    className="btn btn-info",
                    style=CUSTOM_CSS['button']
                ),
                width=4
            ),
            dbc.Col(
                html.A(
                    "Download Full Matrix",
                    id='download-full-link',
                    href='',
                    download='full_matrix.csv',
                    className="btn btn-success",
                    style=CUSTOM_CSS['button']
                ),
                width=4
            ),
            dbc.Col(
                html.A(
                    "Download Subset Matrix",
                    id='download-subset-link',
                    href='',
                    download='subset_matrix.csv',
                    className="btn btn-warning",
                    style=CUSTOM_CSS['button']
                ),
                width=4
            )
        ], style={'margin-top': '20px'}, className="mb-4"),
        dbc.Row(
            dbc.Col(
                html.Div(id='loading-output', style={'margin-top': '20px'})
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Img(id='heatmap-image', src='', style=CUSTOM_CSS['image'])
            )
        ),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='subset-dropdown',
                    options=[],
                    multi=True,
                    placeholder="Select lines for subset matrix...",
                    style=CUSTOM_CSS['dropdown']
                )
            )
        ], style={'margin-top': '20px'}),
        html.Div(id='full-matrix-csv-path', style={'display': 'none'}),
        dcc.Store(id='matrix-store')
    ]
    return base_layout(content)


def progeny_finder_layout():
    """Layout for the progeny finder page."""
    content = [
        dbc.Row(dbc.Col(dcc.Dropdown(
            id='progeny-module-dropdown',
            options=[
                {'label': 'Lookup Progeny of Specific Pairing', 'value': 'specific-pairing'},
                {'label': 'Lookup Progeny of Single Parent', 'value': 'single-parent'},
                {'label': 'Generate Family Tree', 'value': 'family-tree'},
                {'label': 'Generate Descendant Tree', 'value': 'descendant-tree'},
                {'label': 'Generate Combined Family Tree for Two Lines', 'value': 'combined-family-tree'},
                {'label': 'Generate Family Tree with Temporary Progeny', 'value': 'temp-progeny-tree'}
            ],
            multi=True,
            placeholder="Select up to two functions",
            style=CUSTOM_CSS['dropdown']
        ))),
        html.Div(id='selected-progeny-modules', children=[]),
        dbc.Row(dbc.Col(html.A("Back to Main Page", href="/main-page",
                               className="btn btn-secondary", style=CUSTOM_CSS['button']))),
    ]
    return base_layout(content)


def dummy_progeny_matrix_layout():
    """Layout for the dummy progeny matrix page."""
    content = [
        dbc.Row(dbc.Col(html.H3("Add Pedigree Entries", className="app-header", style=CUSTOM_CSS['header']))),
        dbc.Row([
            dbc.Col(dcc.Input(
                id='dummy-progeny-name',
                type='text',
                placeholder="Enter Dummy Progeny Name",
                style={'marginBottom': '10px'}
            )),
            dbc.Col(dcc.Dropdown(
                id='temp-female-parent-dropdown-main',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select Female Parent",
                style=CUSTOM_CSS['dropdown']
            )),
            dbc.Col(dcc.Dropdown(
                id='temp-male-parent-dropdown-main',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select Male Parent",
                style=CUSTOM_CSS['dropdown']
            )),
            dbc.Col(html.Button("Add Dummy Progeny", id="add-temp-progeny-main-button",
                                className="btn btn-warning", style=CUSTOM_CSS['button'])),
            dbc.Col(html.Button("Reset Dummy Progeny", id="reset-temp-progeny-main-button",
                                className="btn btn-danger", style=CUSTOM_CSS['button']))
        ]),
        dbc.Row(dbc.Col(html.Div(id='temp-progeny-list-main'))),
        dbc.Row([
            dbc.Col(html.Button("Get Matrix", id="generate-amatrix-button-dummy",
                                className="btn btn-info", style=CUSTOM_CSS['button']),
                    width=3),
            dbc.Col(html.A("Download Full Matrix", id='download-full-link-dummy',
                           href='', download='full_matrix.csv',
                           className="btn btn-success", style=CUSTOM_CSS['button'])),
            dbc.Col(html.A("Download Subset Matrix", id='download-subset-link-dummy',
                           href='', download='subset_matrix.csv',
                           className="btn btn-warning", style=CUSTOM_CSS['button']))
        ]),
        dbc.Row(dbc.Col(html.Div(id='loading-output-dummy'))),
        dbc.Row(dbc.Col(html.Img(id='heatmap-image-dummy', src='', style=CUSTOM_CSS['image']))),
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='subset-dropdown-dummy',
                options=[],
                multi=True,
                placeholder="Subset A-Matrix",
                style=CUSTOM_CSS['dropdown']
            ))
        ]),
        html.Div(id='full-matrix-csv-path-dummy', style={'display': 'none'}),
        dcc.Store(id='matrix-store-dummy')
    ]
    return base_layout(content)


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


# -------------------------
# Page Navigation Callbacks
# -------------------------
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    """Displays the appropriate page based on the URL path."""
    if pathname == '/main-page':
        return main_page_layout()
    elif pathname == '/progeny-finder':
        return progeny_finder_layout()
    elif pathname == '/dummy-progeny-matrix':
        return dummy_progeny_matrix_layout()
    elif pathname == '/upload':
        return upload_page_layout()
    else:
        return splash_page_layout()


@app.callback(
    Output('info-modal', 'is_open'),
    [Input('open-info-modal', 'n_clicks'), Input('close-info-modal', 'n_clicks')],
    [State('info-modal', 'is_open')]
)
def toggle_info_modal(n1, n2, is_open):
    """Toggles the information modal."""
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('url', 'pathname'),
    [Input('home-button', 'n_clicks')],
    prevent_initial_call=True
)
def navigate_pages(home_clicks):
    """Navigates to the appropriate page."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'home-button':
        return '/'
    return dash.no_update


# =============================================================================
# 3. Callback to Clear Data (reset df to default_df)
# =============================================================================
@app.callback(
    Output('clear-data-button', 'children'),
    Input('clear-data-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_uploaded_data(n_clicks):
    """
    Resets the global df to the default built-in sugarcane data
    when the trash button is clicked.
    """
    if n_clicks:
        global df, filtered_df, parents_set

        df = default_df.copy()
        parents_set = set(df['MaleParent'].tolist() + df['FemaleParent'].tolist())
        mask = (
            ~df['LineName'].isin(parents_set)
            & (df['MaleParent'].isna() | df['MaleParent'].isnull())
            & (df['FemaleParent'].isna() | df['FemaleParent'].isnull())
        )
        filtered_df = df[~mask]

        # Update the button's children to reflect success (optional)
        return [html.I(className="fas fa-trash"), " Cleared!"]
    return dash.no_update


# -----------------------
# Splash Data Modal Stuff
# -----------------------
@app.callback(
    Output('splash-upload-section', 'style'),
    Input('splash-data-choice', 'value')
)
def toggle_splash_upload_section(choice):
    if choice == 'upload':
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    Output('splash-upload-status', 'children'),
    [Input('splash-upload-data', 'contents')],
    [State('splash-upload-data', 'filename')]
)
def upload_file_splash(contents, filename):
    if contents is None:
        return ""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith('.txt'):
            uploaded_df = pd.read_csv(BytesIO(decoded), sep='\t')
        elif filename.lower().endswith('.csv'):
            uploaded_df = pd.read_csv(BytesIO(decoded))
        else:
            return "Unsupported file format (must be .csv or .txt)."

        global df, filtered_df, parents_set
        df = uploaded_df
        parents_set = set(df['MaleParent'].tolist() + df['FemaleParent'].tolist())
        mask = (
            ~df['LineName'].isin(parents_set)
            & (df['MaleParent'].isna() | df['MaleParent'].isnull())
            & (df['FemaleParent'].isna() | df['FemaleParent'].isnull())
        )
        filtered_df = df[~mask]

        return f"Successfully uploaded {filename}"
    except Exception as e:
        print(e)
        return "Error processing file"


@app.callback(
    Output('splash-data-modal', 'is_open'),
    [Input('splash-modal-proceed-btn', 'n_clicks')],
    [State('splash-data-modal', 'is_open')]
)
def close_splash_modal(n_clicks, is_open):
    if n_clicks:
        return False
    return is_open


# ------------------------------------
# Pedigree / Family Tree Helper Funcs
# ------------------------------------
def find_ancestors(line_name, df, ancestors=None, relationships=None, generation=0, generations=None):
    if ancestors is None:
        ancestors = set()
    if relationships is None:
        relationships = []
    if generations is None:
        generations = defaultdict(list)

    current_line = df[df['LineName'] == line_name]
    if not current_line.empty:
        male_parent = current_line.iloc[0]['MaleParent']
        female_parent = current_line.iloc[0]['FemaleParent']

        if pd.notna(male_parent) and not df[df['LineName'] == male_parent].empty:
            if (male_parent, line_name, 'male') not in relationships:
                relationships.append((male_parent, line_name, 'male'))
            if male_parent not in ancestors:
                ancestors.add(male_parent)
                generations[generation + 1].append(male_parent)
                find_ancestors(male_parent, df, ancestors, relationships, generation + 1, generations)

        if pd.notna(female_parent) and not df[df['LineName'] == female_parent].empty:
            if (female_parent, line_name, 'female') not in relationships:
                relationships.append((female_parent, line_name, 'female'))
            if female_parent not in ancestors:
                ancestors.add(female_parent)
                generations[generation + 1].append(female_parent)
                find_ancestors(female_parent, df, ancestors, relationships, generation + 1, generations)
    else:
        # If there's no row for line_name, just add it to ancestors anyway
        ancestors.add(line_name)

    generations[generation].append(line_name)
    return ancestors, relationships, generations


def find_descendants(line_name, df, descendants=None, relationships=None, generation=0, generations=None):
    if descendants is None:
        descendants = set()
    if relationships is None:
        relationships = []
    if generations is None:
        generations = defaultdict(list)

    subset_male = df[df['MaleParent'] == line_name]
    subset_female = df[df['FemaleParent'] == line_name]
    current_line = pd.concat([subset_male, subset_female])

    if not current_line.empty:
        for _, row in current_line.iterrows():
            child = row['LineName']
            if (line_name, child, 'descendant') not in relationships:
                relationships.append((line_name, child, 'descendant'))
            if child not in descendants:
                descendants.add(child)
                generations[generation + 1].append(child)
                find_descendants(child, df, descendants, relationships, generation + 1, generations)

    generations[generation].append(line_name)
    return descendants, relationships, generations


def find_relatives(line_names, df):
    # Gather all ancestors + all descendants for each line_name
    all_ancestors = set()
    all_relationships = []
    generations = defaultdict(list)
    for ln in line_names:
        ans, rels, gens = find_ancestors(ln, df)
        all_ancestors.update(ans)
        all_relationships.extend(rels)
        for g, lines in gens.items():
            generations[g].extend(lines)

    all_descendants = set()
    for ln in line_names:
        desc, rels, gens = find_descendants(ln, df)
        all_descendants.update(desc)
        all_relationships.extend(rels)
        for g, lines in gens.items():
            generations[g].extend(lines)

    # Combine the sets + original lines themselves
    all_relatives = list(all_ancestors.union(all_descendants).union(line_names))
    return all_relatives, all_relationships, generations


def get_maternal_line(line_name, df, collected=None):
    if collected is None:
        collected = []
    row = df[df['LineName'] == line_name]
    if not row.empty:
        female_parent = row.iloc[0]['FemaleParent']
        if pd.notna(female_parent) and female_parent != '':
            collected.append(female_parent)
            get_maternal_line(female_parent, df, collected)
    return collected


def get_paternal_line(line_name, df, collected=None):
    if collected is None:
        collected = []
    row = df[df['LineName'] == line_name]
    if not row.empty:
        male_parent = row.iloc[0]['MaleParent']
        if pd.notna(male_parent) and male_parent != '':
            collected.append(male_parent)
            get_paternal_line(male_parent, df, collected)
    return collected


def sort_pedigree_df(pedigree_df):
    """
    Sorts the pedigree DataFrame in topological order so that each parent
    appears before its progeny. This is essential for a single-pass matrix build.
    """
    sorted_df = pd.DataFrame(columns=pedigree_df.columns)
    processed = set()

    def add_to_sorted_df(line_name):
        if line_name in processed:
            return
        row = pedigree_df[pedigree_df['LineName'] == line_name]
        if not row.empty:
            sire = row.iloc[0]['MaleParent']
            dam = row.iloc[0]['FemaleParent']

            # First recurse on parents
            if pd.notna(sire) and sire in pedigree_df['LineName'].values:
                add_to_sorted_df(sire)
            if pd.notna(dam) and dam in pedigree_df['LineName'].values:
                add_to_sorted_df(dam)

            # Now add this row
            nonlocal sorted_df
            sorted_df = pd.concat([sorted_df, row], ignore_index=True)
            processed.add(line_name)

    # Recurse over all lines
    for ln in pedigree_df['LineName']:
        add_to_sorted_df(ln)

    return sorted_df

# ==================
# NEW: Numba JIT
# ==================
@njit
def _build_matrix_numba(n, sire_idxs, dam_idxs):
    """
    Core Henderson logic, compiled by Numba for speed.
    sire_idxs[i], dam_idxs[i] = the row indices of the sire and dam for individual i,
    or -1 if unknown.
    """
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        s_i = sire_idxs[i]
        d_i = dam_idxs[i]
        if s_i == -1 and d_i == -1:
            # No known parents
            A[i, i] = 1.0
        elif s_i != -1 and d_i == -1:
            # Only sire known
            A[i, i] = 1.0
            A[i, :i] = 0.5 * A[s_i, :i]
            A[:i, i] = A[i, :i]
        elif s_i == -1 and d_i != -1:
            # Only dam known
            A[i, i] = 1.0
            A[i, :i] = 0.5 * A[d_i, :i]
            A[:i, i] = A[i, :i]
        else:
            # Both parents known
            A[i, i] = 1.0 + 0.5 * A[s_i, d_i]
            temp = 0.5 * (A[s_i, :i] + A[d_i, :i])
            A[i, :i] = temp
            A[:i, i] = temp
    return A


def compute_amatrix_diploid_revised(pedigree_df):
    """
    Builds the additive relationship matrix (A-matrix) using a single-pass,
    topologically sorted method (Henderson's approach) for diploid organisms,
    with a Numba-compiled core for speed.
    """
    # 1. Topologically sort the pedigree so parents come before children.
    sorted_df = sort_pedigree_df(pedigree_df)

    # 2. Extract the sorted list of individuals.
    individuals = sorted_df['LineName'].tolist()
    idx_map = {ind: i for i, ind in enumerate(individuals)}
    n = len(individuals)

    # Build arrays of sire, dam row indexes
    sire_idxs = np.full(n, -1, dtype=np.int64)
    dam_idxs = np.full(n, -1, dtype=np.int64)

    for i in range(n):
        row = sorted_df.iloc[i]
        sire = row['MaleParent']
        dam = row['FemaleParent']
        if pd.notna(sire) and sire in idx_map:
            sire_idxs[i] = idx_map[sire]
        if pd.notna(dam) and dam in idx_map:
            dam_idxs[i] = idx_map[dam]

    # 3. Use the JIT-compiled method to build the matrix
    A = _build_matrix_numba(n, sire_idxs, dam_idxs)

    # Convert to a labeled DataFrame
    A_df = pd.DataFrame(A, index=individuals, columns=individuals)
    return A_df


# -------------------
# Descendant Tree CB
# -------------------
@app.callback(
    Output('descendant-tree-image', 'children'),
    [Input('generate-descendant-tree-button', 'n_clicks')],
    [State('descendant-tree-dropdown', 'value')]
)
def generate_descendant_tree(n_clicks, selected_line_name):
    if n_clicks is None or not selected_line_name:
        return dash.no_update

    all_descendants, relationships, generations = find_descendants(selected_line_name, filtered_df)
    subset_descendants = set()
    subset_relationships = []

    # Collect descendants by generation, no slider here so we take all.
    for gen in range(max(generations.keys()) + 1):
        subset_descendants.update(generations[gen])

    for parent, child, role in relationships:
        if parent in subset_descendants and child in subset_descendants:
            subset_relationships.append((parent, child, role))

    dot = graphviz.Digraph(comment='Descendant Tree')
    dot.attr('node', shape='ellipse', style='filled')

    for descendant in subset_descendants:
        label = f"{descendant}"
        node_color = 'lightgrey'
        if descendant == selected_line_name:
            node_color = 'green'  # highlight the root line for clarity
        dot.node(descendant, label=label, fillcolor=node_color, fontcolor='black', color='black')

    for parent, child, role in subset_relationships:
        dot.edge(parent, child, color='black')

    # Convert to PNG
    tree_buffer = BytesIO()
    tree_buffer.write(dot.pipe(format='png'))
    tree_buffer.seek(0)
    encoded_tree = base64.b64encode(tree_buffer.read()).decode('utf-8')
    tree_src = f'data:image/png;base64,{encoded_tree}'
    return html.Img(src=tree_src, style=CUSTOM_CSS['image'])


# -------------------------
# Family Tree Slider & CBs
# -------------------------
@app.callback(
    [Output('generation-depth-slider', 'max'),
     Output('generation-depth-slider', 'marks'),
     Output('generation-depth-slider', 'value')],
    [Input('family-tree-dropdown', 'value')]
)
def update_generation_depth_slider(selected_line_name):
    """Sets the range for the generation depth slider based on max gen found."""
    if not selected_line_name:
        return 0, {}, 0

    _, _, generations = find_ancestors(selected_line_name, filtered_df)
    if generations:
        max_generation = max(generations.keys())
    else:
        max_generation = 0

    marks = {i: str(i) for i in range(max_generation + 1)}
    return max_generation, marks, max_generation


@app.callback(
    Output('family-tree-image', 'children'),
    [
        Input('generate-family-tree-button', 'n_clicks'),
        Input('lineage-highlight-radio', 'value')
    ],
    [
        State('family-tree-dropdown', 'value'),
        State('generation-depth-slider', 'value')
    ]
)
def generate_family_tree(n_clicks, lineage_mode, selected_line_name, generation_depth):
    """
    Generates a family tree that can highlight maternal/paternal lines or color by kinship.
    """
    if not n_clicks or not selected_line_name:
        return dash.no_update

    # 1. Gather all ancestors + relationships for the selected line.
    all_ancestors, relationships, generations = find_ancestors(selected_line_name, filtered_df)

    # 2. Subset only ancestors up to `generation_depth`.
    subset_ancestors = set()
    for gen in range(generation_depth + 1):
        subset_ancestors.update(generations.get(gen, []))

    # 3. Build a subset pedigree df for these ancestors, compute an A-matrix to get kinship.
    relatives_df = filtered_df[filtered_df['LineName'].isin(subset_ancestors)]
    sorted_relatives_df = sort_pedigree_df(relatives_df)
    subset_amatrix = compute_amatrix_diploid_revised(sorted_relatives_df)

    # If the selected line isn't in the matrix, just display a trivial graph.
    if selected_line_name not in subset_amatrix.index:
        dot = graphviz.Digraph(comment='Family Tree')
        dot.node(selected_line_name)
        tree_buf = BytesIO()
        tree_buf.write(dot.pipe(format='png'))
        tree_buf.seek(0)
        enc_tree = base64.b64encode(tree_buf.read()).decode('utf-8')
        return html.Img(src=f'data:image/png;base64,{enc_tree}', style=CUSTOM_CSS['image'])

    # Extract kinship values for color shading (if lineage_mode=='none').
    kinship_values = {
        line: subset_amatrix.loc[selected_line_name, line]
        for line in subset_amatrix.index
    }

    # Determine maternal/paternal sets if highlighting.
    maternal_line = set()
    paternal_line = set()
    if lineage_mode in ['maternal', 'both']:
        maternal_line = set(get_maternal_line(selected_line_name, filtered_df))
    if lineage_mode in ['paternal', 'both']:
        paternal_line = set(get_paternal_line(selected_line_name, filtered_df))

    # Filter relationships to only those in the subset.
    subset_relationships = []
    for parent, child, role in relationships:
        if (parent in subset_ancestors) and (child in subset_ancestors):
            subset_relationships.append((parent, child, role))

    # Color logic: 'none' => color by kinship gradient, else highlight lines.
    if lineage_mode == 'none':
        # color by kinship gradient
        kin_vals = list(kinship_values.values())
        norm = Normalize(vmin=min(kin_vals), vmax=max(kin_vals))
        cmap = cm.get_cmap('Spectral')

        def get_node_color(kval):
            color = cmap(norm(kval))
            return '#{:02x}{:02x}{:02x}'.format(
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255)
            )
    else:
        # highlight maternal/paternal lines with pink/blue/yellow/white
        def get_node_color(line_name):
            if line_name in maternal_line and line_name in paternal_line:
                return '#FFFF99'  # pale yellow for both lines
            elif line_name in maternal_line:
                return '#FFC0CB'  # pink for maternal
            elif line_name in paternal_line:
                return '#ADD8E6'  # light blue for paternal
            elif line_name == selected_line_name:
                return 'lightgreen'  # highlight main line as well
            else:
                return 'white'

    # Build graphviz digraph
    dot = graphviz.Digraph(comment='Family Tree')
    dot.attr('node', shape='ellipse', style='filled', fontsize='20')

    for line in subset_ancestors:
        kv = kinship_values.get(line, 0.0)
        label = f"{line}\nKinship: {kv:.2f}"

        if lineage_mode == 'none':
            node_color = get_node_color(kv)
        else:
            node_color = get_node_color(line)

        dot.node(line, label=label, fillcolor=node_color, fontcolor='black', color='black')

    for parent, child, role in subset_relationships:
        if role == 'male':
            edge_color = 'blue'
        elif role == 'female':
            edge_color = 'red'
        else:
            edge_color = 'black'
        dot.edge(parent, child, color=edge_color)

    # Convert to PNG image
    tree_buffer = BytesIO()
    tree_buffer.write(dot.pipe(format='png'))
    tree_buffer.seek(0)
    encoded_tree = base64.b64encode(tree_buffer.read()).decode('utf-8')
    tree_src = f'data:image/png;base64,{encoded_tree}'
    return html.Img(src=tree_src, style=CUSTOM_CSS['image'])


# ----------------------
# line-name-dropdown CB
# ----------------------
@app.callback(
    Output('line-name-dropdown', 'value'),
    [Input('line-name-dropdown', 'value'),
     Input('paste-line-names', 'value')]
)
def update_line_selection(selected_lines, pasted_lines):
    """
    Parses comma-separated input pasted into the input box and adds valid lines to the selection.
    """
    if pasted_lines:
        pasted_list = [name.strip() for name in pasted_lines.split(',')]
        all_lines = df['LineName'].unique()
        valid_lines = [line for line in pasted_list if line in all_lines]

        if selected_lines:
            # Combine with existing selections without duplicates
            valid_lines = list(set(selected_lines + valid_lines))

        return valid_lines

    return selected_lines


@app.callback(
    Output('selected-line-names-list', 'children'),
    [Input('line-name-dropdown', 'value')]
)
def update_selected_line_names_list(selected_line_names):
    if not selected_line_names:
        return "No lines selected."
    return [html.Li(name) for name in selected_line_names]


# -------------------------
# "Old" relative-finding methods (to gather lines) for the main matrix generation
# -------------------------
def old_find_ancestors(line_names, df, processed=None):
    if processed is None:
        processed = set()
    parents_df = df[df['LineName'].isin(line_names)][['MaleParent', 'FemaleParent']].dropna()
    parents = parents_df['MaleParent'].tolist() + parents_df['FemaleParent'].tolist()
    new_parents = [p for p in parents if p not in processed]

    if not new_parents:
        return line_names

    processed.update(new_parents)
    all_relatives = list(set(line_names + new_parents))
    return old_find_ancestors(all_relatives, df, processed)


def old_find_descendants(line_names, df, processed=None):
    if processed is None:
        processed = set()
    # Must use OR condition for either male/female parent matching
    progeny_df = df[df['MaleParent'].isin(line_names) | df['FemaleParent'].isin(line_names)]
    progeny = progeny_df['LineName'].tolist()
    new_progeny = [child for child in progeny if child not in processed]

    if not new_progeny:
        return line_names

    processed.update(new_progeny)
    all_relatives = list(set(line_names + new_progeny))
    return old_find_descendants(all_relatives, df, processed)


def old_find_relatives(line_names, df):
    # Finds all ancestors + all descendants for the specified line(s)
    ancestors = old_find_ancestors(line_names, df)
    descendants = old_find_descendants(line_names, df)
    # Also gather ancestors of the descendants themselves (so we get a fuller set)
    descendant_ancestors = [old_find_ancestors([d], df) for d in descendants]
    flat_desc_ances = [item for sublist in descendant_ancestors for item in sublist]
    all_relatives = list(set(ancestors + descendants + flat_desc_ances))
    return all_relatives


# -------------------------
# Generating the A-matrix & Heatmap for MAIN PAGE
# -------------------------
@app.callback(
    [
        Output('heatmap-image', 'src'),
        Output('download-full-link', 'href'),
        Output('full-matrix-csv-path', 'children'),
        Output('matrix-store', 'data')
    ],
    [Input('generate-amatrix-button', 'n_clicks')],
    [State('line-name-dropdown', 'value')]
)
def generate_amatrix_and_heatmap(n_clicks, selected_line_names):
    global current_amatrix

    if n_clicks is None or not selected_line_names:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Gather all relatives for selected lines.
    all_related_lines = old_find_relatives(selected_line_names, filtered_df)
    # Filter the DataFrame to only those lines.
    relatives_df = filtered_df[filtered_df['LineName'].isin(all_related_lines)]

    # Compute A-matrix (now done in a single pass with topological sort + numba).
    current_amatrix = compute_amatrix_diploid_revised(relatives_df)

    # Write full matrix to a temporary file.
    temp_file_full = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    current_amatrix.to_csv(temp_file_full)
    temp_file_full.close()

    # ---- NEW: skip or reduce clustering if the matrix is too large
    n_lines = len(current_amatrix)
    if n_lines == 0:
        # Edge case: no lines
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if n_lines <= MAX_CLUSTER_SIZE:
        # Use clustermap
        heatmap_plot = sns.clustermap(
            current_amatrix,
            method='average',
            cmap='Spectral',
            figsize=(15, 15),
            row_cluster=True,
            col_cluster=True
        )
        plt.close(heatmap_plot.fig)

        heatmap_buffer = BytesIO()
        heatmap_plot.savefig(heatmap_buffer, format='png')
        heatmap_buffer.seek(0)
        encoded_heatmap = base64.b64encode(heatmap_buffer.read()).decode('utf-8')
        heatmap_src = f'data:image/png;base64,{encoded_heatmap}'

    else:
        # If too big, do a simpler heatmap (no hierarchical clustering).
        plt.figure(figsize=(15, 15))
        sns.heatmap(current_amatrix, cmap='Spectral')
        # We can also add a note in the corner that clustering was skipped, etc.
        plt.title(f"Heatmap (No Clustering) for {n_lines} lines > {MAX_CLUSTER_SIZE}")
        heatmap_buffer = BytesIO()
        plt.savefig(heatmap_buffer, format='png')
        heatmap_buffer.seek(0)
        plt.close()
        encoded_heatmap = base64.b64encode(heatmap_buffer.read()).decode('utf-8')
        heatmap_src = f'data:image/png;base64,{encoded_heatmap}'

    # Build link for full matrix download.
    full_matrix_link = f'/download?filename={temp_file_full.name}&type=full'
    store_data = current_amatrix.to_dict()
    return (
        heatmap_src,
        full_matrix_link,
        temp_file_full.name,  # hidden store of the path
        store_data
    )


# -------------------------
# Generating the A-matrix & Heatmap for DUMMY PROGENY PAGE
# -------------------------
@app.callback(
    [
        Output('heatmap-image-dummy', 'src'),
        Output('download-full-link-dummy', 'href'),
        Output('full-matrix-csv-path-dummy', 'children'),
        Output('matrix-store-dummy', 'data')
    ],
    [Input('generate-amatrix-button-dummy', 'n_clicks')],
    [State('temp-progeny-list-main', 'children')]
)
def generate_amatrix_and_heatmap_dummy(n_clicks, temp_progeny_list_children):
    global current_amatrix, temp_progeny_list
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Build a DF that includes filtered_df plus any dummy progeny rows.
    temp_df = filtered_df.copy()
    for dummy_progeny in temp_progeny_list:
        temp_df = pd.concat([temp_df, pd.DataFrame([dummy_progeny], columns=temp_df.columns)])

    # Find relatives for all newly added dummy progenies.
    progeny_names = [p[0] for p in temp_progeny_list]
    all_related_lines = old_find_relatives(progeny_names, temp_df)
    relatives_df = temp_df[temp_df['LineName'].isin(all_related_lines)]

    # Compute A-matrix (single-pass method with numba).
    current_amatrix = compute_amatrix_diploid_revised(relatives_df)

    # Temporary file for full matrix.
    temp_file_full = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    current_amatrix.to_csv(temp_file_full)
    temp_file_full.close()

    n_lines = len(current_amatrix)
    if n_lines == 0:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if n_lines <= MAX_CLUSTER_SIZE:
        heatmap_plot = sns.clustermap(
            current_amatrix,
            method='average',
            cmap='Spectral',
            figsize=(15, 15),
            row_cluster=True,
            col_cluster=True
        )
        plt.close(heatmap_plot.fig)

        heatmap_buffer = BytesIO()
        heatmap_plot.savefig(heatmap_buffer, format='png')
        heatmap_buffer.seek(0)
        encoded_heatmap = base64.b64encode(heatmap_buffer.read()).decode('utf-8')
        heatmap_src = f'data:image/png;base64,{encoded_heatmap}'
    else:
        plt.figure(figsize=(15, 15))
        sns.heatmap(current_amatrix, cmap='Spectral')
        plt.title(f"Heatmap (No Clustering) for {n_lines} lines > {MAX_CLUSTER_SIZE}")
        heatmap_buffer = BytesIO()
        plt.savefig(heatmap_buffer, format='png')
        heatmap_buffer.seek(0)
        plt.close()
        encoded_heatmap = base64.b64encode(heatmap_buffer.read()).decode('utf-8')
        heatmap_src = f'data:image/png;base64,{encoded_heatmap}'

    full_matrix_link = f'/download?filename={temp_file_full.name}&type=full'
    store_data = current_amatrix.to_dict()
    return heatmap_src, full_matrix_link, temp_file_full.name, store_data


# -------------------------
# Subset Dropdown for MAIN PAGE & DUMMY PAGE
# -------------------------
@app.callback(
    Output('subset-dropdown', 'options'),
    [Input('generate-amatrix-button', 'n_clicks')],
    [State('matrix-store', 'data')]
)
def update_subset_dropdown_options(n_clicks, matrix_data):
    if n_clicks is None or matrix_data is None:
        return []
    current_amatrix = pd.DataFrame(matrix_data)
    line_names = current_amatrix.index.tolist()
    return [{'label': ln, 'value': ln} for ln in line_names]


@app.callback(
    Output('subset-dropdown-dummy', 'options'),
    [Input('generate-amatrix-button-dummy', 'n_clicks')],
    [State('matrix-store-dummy', 'data')]
)
def update_subset_dropdown_options_dummy(n_clicks, matrix_data):
    if n_clicks is None or matrix_data is None:
        return []
    current_amatrix = pd.DataFrame(matrix_data)
    line_names = current_amatrix.index.tolist()
    return [{'label': ln, 'value': ln} for ln in line_names]


@app.callback(
    Output('subset-matrix-display', 'children'),
    [Input('subset-dropdown', 'value')],
    [State('matrix-store', 'data')]
)
def display_subset_matrix(subset_values, matrix_data):
    if not subset_values or not matrix_data:
        return ""
    current_amatrix = pd.DataFrame(matrix_data)
    subset_amatrix = current_amatrix.loc[subset_values, subset_values]

    table_header = [html.Thead(html.Tr([html.Th('')] + [html.Th(col) for col in subset_amatrix.columns]))]
    table_body = [html.Tbody([
        html.Tr([html.Td(subset_amatrix.index[i])] +
                [html.Td(subset_amatrix.iloc[i, j]) for j in range(len(subset_amatrix.columns))])
        for i in range(len(subset_amatrix))
    ])]
    return html.Table(table_header + table_body, className="table")


@app.callback(
    Output('subset-matrix-display-dummy', 'children'),
    [Input('subset-dropdown-dummy', 'value')],
    [State('matrix-store-dummy', 'data')]
)
def display_subset_matrix_dummy(subset_values, matrix_data):
    if not subset_values or not matrix_data:
        return ""
    current_amatrix = pd.DataFrame(matrix_data)
    subset_amatrix = current_amatrix.loc[subset_values, subset_values]

    table_header = [html.Thead(html.Tr([html.Th('')] + [html.Th(col) for col in subset_amatrix.columns]))]
    table_body = [html.Tbody([
        html.Tr([html.Td(subset_amatrix.index[i])] +
                [html.Td(subset_amatrix.iloc[i, j]) for j in range(len(subset_amatrix.columns))])
        for i in range(len(subset_amatrix))
    ])]
    return html.Table(table_header + table_body, className="table")


# -------------------------
# Download Endpoints
# -------------------------
@app.server.route('/download')
def download_file():
    """Handles file download requests from the generated CSV tempfiles."""
    filename = flask.request.args.get('filename')
    matrix_type = flask.request.args.get('type', 'full')
    if filename:
        return send_file(
            filename,
            mimetype="text/csv",
            as_attachment=True,
            attachment_filename=f'{matrix_type}_matrix.csv'
        )
    else:
        return "File not found", 404


@app.callback(
    Output('download-subset-link', 'href'),
    [Input('subset-dropdown', 'value')]
)
def update_subset_matrix_download_link(subset_values):
    if not subset_values or current_amatrix is None:
        return dash.no_update
    subset_amatrix = current_amatrix.loc[subset_values, subset_values]
    temp_file_subset = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    subset_amatrix.to_csv(temp_file_subset)
    temp_file_subset.close()

    return f'/download?filename={temp_file_subset.name}&type=subset'


@app.callback(
    Output('download-subset-link-dummy', 'href'),
    [Input('subset-dropdown-dummy', 'value')]
)
def update_subset_matrix_download_link_dummy(subset_values):
    if not subset_values or current_amatrix is None:
        return dash.no_update
    subset_amatrix = current_amatrix.loc[subset_values, subset_values]
    temp_file_subset = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    subset_amatrix.to_csv(temp_file_subset)
    temp_file_subset.close()

    return f'/download?filename={temp_file_subset.name}&type=subset'


# -------------------------
# Progeny Finder Callbacks
# -------------------------
@app.callback(
    Output('selected-progeny-modules', 'children'),
    [Input('progeny-module-dropdown', 'value')]
)
def display_selected_progeny_modules(selected_functions):
    """Dynamically display the relevant input sections for the selected function(s)."""
    if not selected_functions:
        return []

    modules = []
    # specific-pairing
    if 'specific-pairing' in selected_functions:
        modules.append(html.Div([
            html.H4("Lookup Progeny of Specific Pairing"),
            dcc.Dropdown(
                id='progeny-line-dropdown',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select First Line Name",
                style=CUSTOM_CSS['dropdown']
            ),
            dcc.Dropdown(
                id='progeny-line-dropdown-2',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select Second Line Name",
                style=CUSTOM_CSS['dropdown']
            ),
            html.Button("Find Progeny", id="find-progeny-button",
                        className="btn btn-success", style=CUSTOM_CSS['button']),
            html.Div(id='progeny-results')
        ]))

    # single-parent
    if 'single-parent' in selected_functions:
        modules.append(html.Div([
            html.H4("Lookup Progeny of Single Parent"),
            dcc.Dropdown(
                id='single-parent-dropdown',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select a Parent",
                style=CUSTOM_CSS['dropdown']
            ),
            html.Button("Find Progeny", id="find-single-parent-progeny-button",
                        className="btn btn-info", style=CUSTOM_CSS['button']),
            html.Div(id='single-parent-progeny-results')
        ]))

    # family-tree
    if 'family-tree' in selected_functions:
        modules.append(html.Div([
            html.H4("Generate Family Tree"),
            dcc.Dropdown(
                id='family-tree-dropdown',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select a Line",
                style=CUSTOM_CSS['dropdown']
            ),
            dcc.Slider(
                id='generation-depth-slider',
                min=0,
                max=0,
                step=1,
                value=0,
                marks={},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            dcc.RadioItems(
                id='lineage-highlight-radio',
                options=[
                    {'label': 'No Highlight', 'value': 'none'},
                    {'label': 'Highlight Maternal Only', 'value': 'maternal'},
                    {'label': 'Highlight Paternal Only', 'value': 'paternal'},
                    {'label': 'Highlight Both', 'value': 'both'}
                ],
                value='none',
                labelStyle={'display': 'block', 'marginTop': '5px'}
            ),
            html.Button("Generate Family Tree", id="generate-family-tree-button",
                        className="btn btn-warning", style=CUSTOM_CSS['button']),
            html.Div(id='family-tree-image')
        ]))

    # descendant-tree
    if 'descendant-tree' in selected_functions:
        modules.append(html.Div([
            html.H4("Generate Descendant Tree"),
            dcc.Dropdown(
                id='descendant-tree-dropdown',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select a Line",
                style=CUSTOM_CSS['dropdown']
            ),
            html.Button("Generate Descendant Tree", id="generate-descendant-tree-button",
                        className="btn btn-warning", style=CUSTOM_CSS['button']),
            html.Div(id='descendant-tree-image')
        ]))

    # combined-family-tree
    if 'combined-family-tree' in selected_functions:
        modules.append(html.Div([
            html.H4("Generate Combined Family Tree for Two Lines"),
            dcc.Dropdown(
                id='combined-family-tree-dropdown-1',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select First Line",
                style=CUSTOM_CSS['dropdown']
            ),
            dcc.Dropdown(
                id='combined-family-tree-dropdown-2',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select Second Line",
                style=CUSTOM_CSS['dropdown']
            ),
            html.Button("Generate Combined Family Tree",
                        id="generate-combined-family-tree-button",
                        className="btn btn-primary", style=CUSTOM_CSS['button']),
            html.Div(id='combined-family-tree-image')
        ]))

    # temp-progeny-tree
    if 'temp-progeny-tree' in selected_functions:
        modules.append(html.Div([
            html.H4("Generate Family Tree with Temporary Progeny"),
            dcc.Dropdown(
                id='temp-female-parent-dropdown',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select Female Parent",
                style=CUSTOM_CSS['dropdown']
            ),
            dcc.Dropdown(
                id='temp-male-parent-dropdown',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select Male Parent",
                style=CUSTOM_CSS['dropdown']
            ),
            html.Button("Generate Family Tree with Temporary Progeny",
                        id="generate-temp-progeny-tree-button",
                        className="btn btn-warning", style=CUSTOM_CSS['button']),
            html.Div(id='temp-progeny-tree-image')
        ]))

    return modules


@app.callback(
    Output('progeny-results', 'children'),
    [Input('find-progeny-button', 'n_clicks')],
    [State('progeny-line-dropdown', 'value'),
     State('progeny-line-dropdown-2', 'value')]
)
def find_progeny(n_clicks, line1, line2):
    if n_clicks is None or not line1 or not line2:
        return dash.no_update

    progeny_df = df[
        ((df['MaleParent'] == line1) & (df['FemaleParent'] == line2)) |
        ((df['MaleParent'] == line2) & (df['FemaleParent'] == line1))
    ]
    if progeny_df.empty:
        return "No progeny found for the selected lines."

    return html.Ul([
        html.Li(
            f"{row['LineName']} (Female: {row['FemaleParent']}, Male: {row['MaleParent']})"
        )
        for _, row in progeny_df.iterrows()
    ])


@app.callback(
    Output('single-parent-progeny-results', 'children'),
    [Input('find-single-parent-progeny-button', 'n_clicks')],
    [State('single-parent-dropdown', 'value')]
)
def find_single_parent_progeny(n_clicks, parent):
    if n_clicks is None or not parent:
        return dash.no_update

    progeny_df = df[
        (df['MaleParent'] == parent) | (df['FemaleParent'] == parent)
    ]
    if progeny_df.empty:
        return "No progeny found for the selected parent."

    results = []
    for _, row in progeny_df.iterrows():
        if row['MaleParent'] == parent:
            male_parent = parent
            female_parent = row['FemaleParent']
        else:
            male_parent = row['MaleParent']
            female_parent = parent

        results.append(
            html.Li(f"{row['LineName']} (Female: {female_parent}, Male: {male_parent})")
        )

    return html.Ul(results)


@app.callback(
    Output('combined-family-tree-image', 'children'),
    [Input('generate-combined-family-tree-button', 'n_clicks')],
    [State('combined-family-tree-dropdown-1', 'value'),
     State('combined-family-tree-dropdown-2', 'value')]
)
def generate_combined_family_tree(n_clicks, line1, line2):
    if n_clicks is None or not line1 or line2 is None:
        return dash.no_update

    ancestors1, relationships1, generations1 = find_ancestors(line1, filtered_df)
    ancestors2, relationships2, generations2 = find_ancestors(line2, filtered_df)
    all_lines = ancestors1.union(ancestors2, {line1, line2})
    all_relationships = relationships1 + relationships2

    # Merge generations (not strictly necessary for drawing, but might help if you add depth).
    generations = defaultdict(list)
    if generations1:
        max_g1 = max(generations1.keys())
    else:
        max_g1 = 0
    if generations2:
        max_g2 = max(generations2.keys())
    else:
        max_g2 = 0
    for g in range(max(max_g1, max_g2) + 1):
        generations[g].extend(generations1.get(g, []) + generations2.get(g, []))

    # Determine color for ancestors unique to line1 vs line2 vs shared.
    line_colors = {}
    for a in ancestors1:
        line_colors[a] = '#ADD8E6'  # Light blue for line1 ancestors.
    for a in ancestors2:
        if a in line_colors:
            line_colors[a] = '#FFFFE0'  # Light yellow if shared
        else:
            line_colors[a] = '#FFB6C1'  # Light pink for line2 ancestors.

    # The lines themselves green.
    line_colors[line1] = 'green'
    line_colors[line2] = 'green'

    relatives_df = filtered_df[filtered_df['LineName'].isin(all_lines)]
    dot = graphviz.Digraph(comment='Combined Family Tree')
    dot.attr('node', shape='ellipse', style='filled')

    for _, row in relatives_df.iterrows():
        ln = row['LineName']
        label = f"{ln}"
        node_color = line_colors.get(ln, 'lightgrey')
        dot.node(ln, label=label, fillcolor=node_color, fontcolor='black', color='black')

    added_edges = set()
    for parent, child, role in all_relationships:
        if (parent, child) not in added_edges:
            # Color the edge based on which side it belongs to.
            if child in ancestors1 or child == line1:
                color = '#ADD8E6'
            elif child in ancestors2 or child == line2:
                color = '#FFB6C1'
            else:
                color = 'black'
            dot.edge(parent, child, color=color)
            added_edges.add((parent, child))

    # Convert to PNG
    tree_buffer = BytesIO()
    tree_buffer.write(dot.pipe(format='png'))
    tree_buffer.seek(0)
    encoded_tree = base64.b64encode(tree_buffer.read()).decode('utf-8')
    tree_src = f'data:image/png;base64,{encoded_tree}'

    return html.Img(src=tree_src, style=CUSTOM_CSS['image'])


@app.callback(
    Output('temp-progeny-tree-image', 'children'),
    [Input('generate-temp-progeny-tree-button', 'n_clicks')],
    [State('temp-female-parent-dropdown', 'value'),
     State('temp-male-parent-dropdown', 'value')]
)
def generate_temp_progeny_tree(n_clicks, female_parent, male_parent):
    if n_clicks is None or not female_parent or not male_parent:
        return dash.no_update

    temp_progeny_name = "Temp_Progeny"
    temp_df = filtered_df.copy()
    temp_df = pd.concat([temp_df, pd.DataFrame([[temp_progeny_name, female_parent, male_parent]],
                                               columns=temp_df.columns)])

    # Get ancestors for the newly added line.
    all_ancestors, relationships, generations = find_ancestors(temp_progeny_name, temp_df)
    all_lines = all_ancestors.union({temp_progeny_name})
    relatives_df = temp_df[temp_df['LineName'].isin(all_lines)]

    kinship_matrix = compute_amatrix_diploid_revised(relatives_df)

    # Determine color by kinship to the new line.
    if temp_progeny_name in kinship_matrix.index:
        kinship_values = kinship_matrix.loc[temp_progeny_name, all_lines].drop(temp_progeny_name)
        if not kinship_values.empty:
            quantiles = kinship_values.quantile([0.25, 0.5, 0.75])
        else:
            quantiles = pd.Series([0, 0, 0], index=[0.25, 0.5, 0.75])
    else:
        quantiles = pd.Series([0, 0, 0], index=[0.25, 0.5, 0.75])

    def get_color(value):
        if value <= quantiles[0.25]:
            return '#ADD8E6'  # Light blue
        elif value <= quantiles[0.5]:
            return '#FFB6C1'  # Light pink
        else:
            return '#FFFFE0'  # Light yellow

    dot = graphviz.Digraph(comment='Family Tree with Temporary Progeny')
    dot.attr('node', shape='ellipse', style='filled', fontsize='20')

    for _, row in relatives_df.iterrows():
        line_name = row['LineName']
        if line_name in kinship_matrix.index and temp_progeny_name in kinship_matrix.index:
            kv = kinship_matrix.loc[temp_progeny_name, line_name]
        else:
            kv = 0.0
        label = f"{line_name}\nKinship: {kv:.2f}"

        if line_name == temp_progeny_name:
            color = 'orange'  # highlight the temp progeny
        else:
            color = get_color(kv)

        dot.node(line_name, label=label, fillcolor=color, fontcolor='black', color='black')

    for parent, child, role in relationships:
        if role == 'male':
            edge_color = 'blue'
        elif role == 'female':
            edge_color = 'red'
        else:
            edge_color = 'black'
        dot.edge(parent, child, color=edge_color)

    tree_buffer = BytesIO()
    tree_buffer.write(dot.pipe(format='png'))
    tree_buffer.seek(0)
    encoded_tree = base64.b64encode(tree_buffer.read()).decode('utf-8')
    tree_src = f'data:image/png;base64,{encoded_tree}'

    return html.Img(src=tree_src, style=CUSTOM_CSS['image'])


@app.callback(
    Output('temp-progeny-list-main', 'children'),
    [Input('add-temp-progeny-main-button', 'n_clicks'),
     Input('reset-temp-progeny-main-button', 'n_clicks')],
    [
        State('dummy-progeny-name', 'value'),
        State('temp-female-parent-dropdown-main', 'value'),
        State('temp-male-parent-dropdown-main', 'value')
    ]
)
def update_temp_progeny_list(add_clicks, reset_clicks, progeny_name, female_parent, male_parent):
    global temp_progeny_list, temp_progeny_count
    ctx = dash.callback_context
    if not ctx.triggered:
        return []

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'reset-temp-progeny-main-button':
        temp_progeny_list = []
        temp_progeny_count = 0
        return []

    if button_id == 'add-temp-progeny-main-button':
        if progeny_name and female_parent and male_parent:
            temp_progeny_list.append((progeny_name, female_parent, male_parent))
            temp_progeny_count += 1

    list_items = [
        html.Li(f"{name} (Female: {female}, Male: {male})")
        for name, female, male in temp_progeny_list
    ]
    return html.Ul(list_items)


# -------------------------
# Upload Page Callback
# -------------------------
@app.callback(Output('upload-status', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def upload_file(contents, filename, last_modified):
    if contents is None:
        return ""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename.lower():
            uploaded_df = pd.read_csv(BytesIO(decoded))
        elif 'xls' in filename.lower():
            uploaded_df = pd.read_excel(BytesIO(decoded))
        else:
            return "Unsupported file format"

        global df, filtered_df, parents_set
        df = uploaded_df
        parents_set = set(df['MaleParent'].tolist() + df['FemaleParent'].tolist())
        mask = (
            ~df['LineName'].isin(parents_set)
            & (df['MaleParent'].isna() | df['MaleParent'].isnull())
            & (df['FemaleParent'].isna() | df['FemaleParent'].isnull())
        )
        filtered_df = df[~mask]
        return f"Successfully uploaded {filename}"
    except Exception as e:
        print(e)
        return "There was an error processing the file"


# -------------------------
# Main App Run
# -------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
