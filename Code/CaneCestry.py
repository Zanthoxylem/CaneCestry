import dash
from dash import html, dcc, Input, Output, State, dash_table, callback_context
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

# Initialize Flask server
server = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('/home/Zanthoxylum2117/Canecestry/PedInf.txt', sep="\t")
parents_set = set(df['MaleParent'].tolist() + df['FemaleParent'].tolist())
mask = ~df['LineName'].isin(parents_set) & (df['MaleParent'].isna() | df['MaleParent'].isnull()) & (df['FemaleParent'].isna() | df['FemaleParent'].isnull())
filtered_df = df[~mask]

# Global variables
current_amatrix = None
in_memory_matrices = {}
temp_progeny_count = 0
temp_progeny_list = []
dummy_progeny_list = []

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE, "https://use.fontawesome.com/releases/v5.10.2/css/all.css"], server=server, suppress_callback_exceptions=True)

# Custom CSS styles
CUSTOM_CSS = {
    'container': {'padding': '20px', 'margin-top': '20px', 'background-color': '#f9f9f9'},
    'header': {'textAlign': 'center', 'padding': '10px', 'color': '#2c3e50', 'font-family': 'Arial', 'font-weight': 'bold', 'font-size': '40px'},
    'button': {'margin': '5px', 'font-weight': 'bold', 'background-color': '#2980b9', 'color': 'white'},
    'table': {'overflowX': 'auto', 'margin-bottom': '20px', 'border': '1px solid #ccc'},
    'image': {'width': '100%', 'padding': '10px'},
    'dropdown': {'font-weight': 'bold', 'color': '#2980b9'},
    'static-button': {'position': 'absolute', 'top': '10px', 'right': '10px', 'border-radius': '50%', 'width': '50px', 'height': '50px', 'font-size': '20px', 'textAlign': 'center', 'lineHeight': '50px', 'background-color': '#2980b9', 'color': 'white', 'border': 'none'},
    'home-button': {'position': 'absolute', 'left': '10px', 'top': '10px', 'border-radius': '50%', 'width': '50px', 'height': '50px', 'font-size': '20px', 'textAlign': 'center', 'lineHeight': '50px', 'background-color': '#2980b9', 'color': 'white', 'border': 'none'}
}

def base_layout(content):
    """Generates the base layout with common structure."""
    return dbc.Container([
        dbc.Row(dbc.Col(html.Button(html.I(className="fas fa-home"), id="home-button", style=CUSTOM_CSS['home-button']))),
        dbc.Row(dbc.Col(html.H1("CaneCestry 2.0", className="app-header", style=CUSTOM_CSS['header']))),
        dbc.Row(dbc.Col(html.Div(id='common-content', children=content))),
        dbc.Row(dbc.Col(html.Div(id='page-specific-content'))),
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
                        html.Li("Subset A-Matrix: Choose a subset of the kinship matrix to view."),
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
    """Layout for the splash page."""
    content = [
        dbc.Row(dbc.Col(html.P("Please select one of the following options to proceed:", style={'textAlign': 'center', 'fontSize': '20px'}))),
        dbc.Row([
            dbc.Col(html.A("Use Built-in Pedigree Data", href="/main-page", className="btn btn-primary", style=CUSTOM_CSS['button']), width={"size": 3, "offset": 3}),
            dbc.Col(html.A("Upload Your Own Pedigree Data", href="/upload", className="btn btn-secondary", style=CUSTOM_CSS['button']), width={"size": 3})
        ])
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
        dbc.Row(dbc.Col(html.P("Please upload your pedigree data file. The accepted format is .CSV. The file should contain the following columns:", style={'marginTop': '10px'}))),
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
        dbc.Row(dbc.Col(html.A("Proceed to Main Page", href="/main-page", className="btn btn-primary", style=CUSTOM_CSS['button']))),
    ]
    return base_layout(content)

def main_page_layout():
    """Layout for the main page."""
    content = [
        dbc.Row(dbc.Col(html.Div(id='input-container', children=[]), className="p-3", style={'background-color': '#eaf2f8'})),

        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='line-name-dropdown',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=True,
                placeholder="Select Line Names",
                style=CUSTOM_CSS['dropdown']
            ), width=12)
        ]),

        dbc.Row(dbc.Col(html.Ul(id="selected-line-names-list", children=[]))),

        dbc.Row([
            dbc.Col(dash_table.DataTable(
                id='parent-info-table',
                columns=[
                    {'name': 'LineName', 'id': 'LineName'},
                    {'name': 'Male Parent', 'id': 'MaleParent'},
                    {'name': 'Female Parent', 'id': 'FemaleParent'}
                ],
                data=[],
                style_table=CUSTOM_CSS['table'],
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'left', 'padding': '10px'}
            ), width=6)
        ], style=CUSTOM_CSS['table']),

        dbc.Row([
            dbc.Col(html.Button("Get Matrix", id="generate-amatrix-button", className="btn btn-info", style=CUSTOM_CSS['button']), width=3),
            dbc.Col(html.A("Download Full Matrix", id='download-full-link', href='', download='full_matrix.csv', className="btn btn-success", style=CUSTOM_CSS['button'])),
            dbc.Col(html.A("Download Subset Matrix", id='download-subset-link', href='', download='subset_matrix.csv', className="btn btn-warning", style=CUSTOM_CSS['button']))
        ]),

        dbc.Row(dbc.Col(html.Div(id='loading-output'))),

        dbc.Row(dbc.Col(html.Img(id='heatmap-image', src='', style=CUSTOM_CSS['image']))),

        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='subset-dropdown',
                options=[],
                multi=True,
                placeholder="Subset A-Matrix",
                style=CUSTOM_CSS['dropdown']
            ))
        ]),

        html.Div(id='full-matrix-csv-path', style={'display': 'none'}),
        dcc.Store(id='matrix-store'),

        dbc.Row(dbc.Col(html.A("Go to Progeny Finder", href="/progeny-finder", className="btn btn-primary"))),
        dbc.Row(dbc.Col(html.A("Go to Dummy Progeny Matrix", href="/dummy-progeny-matrix", className="btn btn-primary", style={'marginTop': '10px'})))
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

        dbc.Row(dbc.Col(html.A("Back to Main Page", href="/main-page", className="btn btn-secondary", style=CUSTOM_CSS['button']))),

        # New feature inputs
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='temp-female-parent-dropdown',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select Female Parent",
                style=CUSTOM_CSS['dropdown']
            )),
            dbc.Col(dcc.Dropdown(
                id='temp-male-parent-dropdown',
                options=[{'label': name, 'value': name} for name in df['LineName'].unique()],
                multi=False,
                placeholder="Select Male Parent",
                style=CUSTOM_CSS['dropdown']
            )),
            dbc.Col(html.Button("Generate Family Tree with Temporary Progeny", id="generate-temp-progeny-tree-button", className="btn btn-warning", style=CUSTOM_CSS['button']))
        ]),
        dbc.Row(dbc.Col(html.Div(id='temp-progeny-tree-image')))
    ]
    return base_layout(content)

def dummy_progeny_matrix_layout():
    """Layout for the dummy progeny matrix page."""
    content = [
        dbc.Row(dbc.Col(html.H3("Add Dummy Progeny", className="app-header", style=CUSTOM_CSS['header']))),

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
            dbc.Col(html.Button("Add Dummy Progeny", id="add-temp-progeny-main-button", className="btn btn-warning", style=CUSTOM_CSS['button'])),
            dbc.Col(html.Button("Reset Dummy Progeny", id="reset-temp-progeny-main-button", className="btn btn-danger", style=CUSTOM_CSS['button']))
        ]),

        dbc.Row(dbc.Col(html.Div(id='temp-progeny-list-main'))),

        dbc.Row([
            dbc.Col(html.Button("Get Matrix", id="generate-amatrix-button-dummy", className="btn btn-info", style=CUSTOM_CSS['button']), width=3),
            dbc.Col(html.A("Download Full Matrix", id='download-full-link-dummy', href='', download='full_matrix.csv', className="btn btn-success", style=CUSTOM_CSS['button'])),
            dbc.Col(html.A("Download Subset Matrix", id='download-subset-link-dummy', href='', download='subset_matrix.csv', className="btn btn-warning", style=CUSTOM_CSS['button']))
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

@app.callback(
    Output('selected-progeny-modules', 'children'),
    [Input('progeny-module-dropdown', 'value')]
)
def display_selected_progeny_modules(selected_functions):
    """Displays selected progeny modules."""
    if not selected_functions:
        return html.Div([])

    modules = []
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
            html.Button("Find Progeny", id="find-progeny-button", className="btn btn-success", style=CUSTOM_CSS['button']),
            html.Div(id='progeny-results')
        ]))
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
            html.Button("Find Progeny", id="find-single-parent-progeny-button", className="btn btn-info", style=CUSTOM_CSS['button']),
            html.Div(id='single-parent-progeny-results')
        ]))
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
            html.Button("Generate Family Tree", id="generate-family-tree-button", className="btn btn-warning", style=CUSTOM_CSS['button']),
            html.Div(id='family-tree-image')
        ]))
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
            html.Button("Generate Descendant Tree", id="generate-descendant-tree-button", className="btn btn-warning", style=CUSTOM_CSS['button']),
            html.Div(id='descendant-tree-image')
        ]))
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
            html.Button("Generate Combined Family Tree", id="generate-combined-family-tree-button", className="btn btn-primary", style=CUSTOM_CSS['button']),
            html.Div(id='combined-family-tree-image')
        ]))
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
            html.Button("Generate Family Tree with Temporary Progeny", id="generate-temp-progeny-tree-button", className="btn btn-warning", style=CUSTOM_CSS['button']),
            html.Div(id='temp-progeny-tree-image')
        ]))

    return modules

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

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






def find_ancestors(line_name, df, ancestors=None, relationships=None):
    """Finds all ancestors of a given line."""
    if ancestors is None:
        ancestors = set()
    if relationships is None:
        relationships = []

    current_line = df[df['LineName'] == line_name]
    if not current_line.empty:
        male_parent = current_line.iloc[0]['MaleParent']
        female_parent = current_line.iloc[0]['FemaleParent']

        if pd.notna(male_parent) and not df[df['LineName'] == male_parent].empty:
            if (male_parent, line_name, 'male') not in relationships:
                relationships.append((male_parent, line_name, 'male'))
            if male_parent not in ancestors:
                ancestors.add(male_parent)
                find_ancestors(male_parent, df, ancestors, relationships)
        if pd.notna(female_parent) and not df[df['LineName'] == female_parent].empty:
            if (female_parent, line_name, 'female') not in relationships:
                relationships.append((female_parent, line_name, 'female'))
            if female_parent not in ancestors:
                ancestors.add(female_parent)
                find_ancestors(female_parent, df, ancestors, relationships)
    else:
        ancestors.add(line_name)
    return ancestors, relationships

def find_descendants(line_name, df, descendants=None, relationships=None):
    """Finds all descendants of a given line."""
    if descendants is None:
        descendants = set()
    if relationships is None:
        relationships = []

    current_line = df[df['MaleParent'] == line_name].append(df[df['FemaleParent'] == line_name])
    if not current_line.empty:
        for _, row in current_line.iterrows():
            child = row['LineName']
            if (line_name, child, 'descendant') not in relationships:
                relationships.append((line_name, child, 'descendant'))
            if child not in descendants:
                descendants.add(child)
                find_descendants(child, df, descendants, relationships)
    return descendants, relationships

def find_relatives(line_names, df):
    """Finds all relatives (ancestors and descendants) of given line names."""
    all_ancestors = set()
    all_relationships = []
    for line_name in line_names:
        ancestors, relationships = find_ancestors(line_name, df)
        all_ancestors.update(ancestors)
        all_relationships.extend(relationships)

    all_descendants = set()
    for line_name in line_names:
        descendants, relationships = find_descendants(line_name, df)
        all_descendants.update(descendants)
        all_relationships.extend(relationships)

    all_relatives = list(all_ancestors.union(all_descendants).union(line_names))
    return all_relatives, all_relationships

@app.callback(
    Output('family-tree-image', 'children'),
    [Input('generate-family-tree-button', 'n_clicks')],
    [State('family-tree-dropdown', 'value')]
)
def generate_family_tree(n_clicks, selected_line_name):
    """Generates and displays the family tree for a selected line."""
    if n_clicks is None or not selected_line_name:
        return dash.no_update

    # Find all ancestors
    all_ancestors, relationships = find_ancestors(selected_line_name, filtered_df)
    all_lines = all_ancestors.union({selected_line_name})
    relatives_df = filtered_df[filtered_df['LineName'].isin(all_lines)]

    # Compute the A-matrix (akin to kinship matrix)
    sorted_relatives_df = sort_pedigree_df(relatives_df)
    kinship_matrix = compute_amatrix_diploid_revised(sorted_relatives_df)

    # Determine quantiles for kinship values
    kinship_values = kinship_matrix.loc[selected_line_name, all_lines].drop(selected_line_name)
    quantiles = kinship_values.quantile([0.25, 0.5, 0.75])

    def get_color(value):
        if value <= quantiles[0.25]:
            return '#d73027'  # Dark red
        elif value <= quantiles[0.5]:
            return '#fc8d59'  # Light red
        elif value <= quantiles[0.75]:
            return '#fee08b'  # Yellow
        else:
            return '#91cf60'  # Green

    dot = graphviz.Digraph(comment='Family Tree')
    dot.attr('node', shape='ellipse', style='filled', fontsize='20')  # Increase font size

    for _, row in relatives_df.iterrows():
        line_name = row['LineName']
        label = f"{line_name}"

        # Determine node color based on kinship value
        if line_name != selected_line_name:
            kinship_value = kinship_matrix.loc[selected_line_name, line_name]
            label += f"\nKinship: {kinship_value:.2f}"
            color = get_color(kinship_value)
        else:
            color = 'green'  # Highlight the selected line

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
    Output('descendant-tree-image', 'children'),
    [Input('generate-descendant-tree-button', 'n_clicks')],
    [State('descendant-tree-dropdown', 'value')]
)
def generate_descendant_tree(n_clicks, selected_line_name):
    """Generates and displays the descendant tree for a selected line."""
    if n_clicks is None or not selected_line_name:
        return dash.no_update

    all_descendants, relationships = find_descendants(selected_line_name, filtered_df)
    all_lines = all_descendants.union({selected_line_name})
    relatives_df = filtered_df[filtered_df['LineName'].isin(all_lines)]

    dot = graphviz.Digraph(comment='Descendant Tree')
    dot.attr('node', shape='ellipse', style='filled')

    for _, row in relatives_df.iterrows():
        line_name = row['LineName']
        label = f"{line_name}"

        # Determine node color based on role
        node_color = 'lightgrey'
        if row['LineName'] == selected_line_name:
            node_color = 'green'  # Highlight the selected line

        dot.node(line_name, label=label, fillcolor=node_color, fontcolor='black', color='black')

    for parent, child, role in relationships:
        dot.edge(parent, child, color='black')

    tree_buffer = BytesIO()
    tree_buffer.write(dot.pipe(format='png'))
    tree_buffer.seek(0)
    encoded_tree = base64.b64encode(tree_buffer.read()).decode('utf-8')
    tree_src = f'data:image/png;base64,{encoded_tree}'

    return html.Img(src=tree_src, style=CUSTOM_CSS['image'])

@app.callback(
    Output('selected-line-names-list', 'children'),
    [Input('line-name-dropdown', 'value')]
)
def update_selected_line_names_list(selected_line_names):
    """Updates the list of selected line names."""
    if not selected_line_names:
        return []

    list_items = [html.Li(name) for name in selected_line_names]
    return list_items

@app.callback(
    Output('parent-info-table', 'data'),
    [Input('line-name-dropdown', 'value')]
)
def update_parent_info_table(selected_line_names):
    """Updates the parent information table."""
    if not selected_line_names:
        return []

    filtered_rows = df[df['LineName'].isin(selected_line_names)][['LineName', 'MaleParent', 'FemaleParent']]
    return filtered_rows.to_dict('records')





@app.callback(
    [Output('heatmap-image', 'src'), Output('download-full-link', 'href'), Output('full-matrix-csv-path', 'children'), Output('matrix-store', 'data')],
    [Input('generate-amatrix-button', 'n_clicks')],
    [State('line-name-dropdown', 'value')]
)
def generate_amatrix_and_heatmap(n_clicks, selected_line_names):
    """Generates the A-matrix and heatmap for selected lines."""
    global current_amatrix
    if n_clicks is None or not selected_line_names:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Find all relatives using the old find_relatives function
    all_related_lines = old_find_relatives(selected_line_names, filtered_df)

    # Filter the dataframe to include only the relevant relatives
    relatives_df = filtered_df[filtered_df['LineName'].isin(all_related_lines)]

    # Sort the pedigree dataframe
    sorted_relatives_df = sort_pedigree_df(relatives_df)

    # Compute the A-matrix using the revised compute_amatrix_diploid_revised function
    current_amatrix = compute_amatrix_diploid_revised(sorted_relatives_df)

    # Create a temporary file to store the full matrix
    temp_file_full = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    current_amatrix.to_csv(temp_file_full)
    temp_file_full.close()

    # Generate the heatmap
    heatmap_plot = sns.clustermap(current_amatrix, method='average', cmap='Spectral', figsize=(15, 15), row_cluster=True, col_cluster=True)
    plt.close(heatmap_plot.fig)

    heatmap_buffer = BytesIO()
    heatmap_plot.savefig(heatmap_buffer, format='png')
    heatmap_buffer.seek(0)
    encoded_heatmap = base64.b64encode(heatmap_buffer.read()).decode('utf-8')
    heatmap_src = f'data:image/png;base64,{encoded_heatmap}'

    # Create a filename and store matrix data
    filename = "_".join(selected_line_names) + ".csv"
    matrix_id = str(uuid.uuid4())
    in_memory_matrices[matrix_id] = {
        "full_matrix": current_amatrix,
        "filename": filename
    }

    full_matrix_link = f'/download?filename={temp_file_full.name}&type=full'
    store_data = current_amatrix.to_dict()
    full_matrix_csv_path = temp_file_full.name

    return heatmap_src, full_matrix_link, full_matrix_csv_path, store_data

@app.callback(
    [Output('heatmap-image-dummy', 'src'), Output('download-full-link-dummy', 'href'), Output('full-matrix-csv-path-dummy', 'children'), Output('matrix-store-dummy', 'data')],
    [Input('generate-amatrix-button-dummy', 'n_clicks')],
    [State('temp-progeny-list-main', 'children')]
)
def generate_amatrix_and_heatmap_dummy(n_clicks, temp_progeny_list_children):
    """Generates the A-matrix and heatmap for dummy progeny."""
    global current_amatrix, temp_progeny_list
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    temp_df = filtered_df.copy()

    for dummy_progeny in temp_progeny_list:
        temp_df = pd.concat([temp_df, pd.DataFrame([dummy_progeny], columns=temp_df.columns)])

    # Find all relatives using the old find_relatives function
    all_related_lines = old_find_relatives([progeny[0] for progeny in temp_progeny_list], temp_df)

    # Filter the dataframe to include only the relevant relatives
    relatives_df = temp_df[temp_df['LineName'].isin(all_related_lines)]

    # Sort the pedigree dataframe
    sorted_relatives_df = sort_pedigree_df(relatives_df)

    # Compute the A-matrix using the revised compute_amatrix_diploid_revised function
    current_amatrix = compute_amatrix_diploid_revised(sorted_relatives_df)

    # Create a temporary file to store the full matrix
    temp_file_full = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    current_amatrix.to_csv(temp_file_full)
    temp_file_full.close()

    # Generate the heatmap
    heatmap_plot = sns.clustermap(current_amatrix, method='average', cmap='Spectral', figsize=(15, 15), row_cluster=True, col_cluster=True)
    plt.close(heatmap_plot.fig)

    heatmap_buffer = BytesIO()
    heatmap_plot.savefig(heatmap_buffer, format='png')
    heatmap_buffer.seek(0)
    encoded_heatmap = base64.b64encode(heatmap_buffer.read()).decode('utf-8')
    heatmap_src = f'data:image/png;base64,{encoded_heatmap}'

    # Create a filename and store matrix data
    filename = "dummy_progeny_matrix.csv"
    matrix_id = str(uuid.uuid4())
    in_memory_matrices[matrix_id] = {
        "full_matrix": current_amatrix,
        "filename": filename
    }

    full_matrix_link = f'/download?filename={temp_file_full.name}&type=full'
    store_data = current_amatrix.to_dict()
    full_matrix_csv_path = temp_file_full.name

    return heatmap_src, full_matrix_link, full_matrix_csv_path, store_data





@app.callback(
    Output('subset-dropdown', 'options'),
    [Input('generate-amatrix-button', 'n_clicks')],
    [State('matrix-store', 'data')]
)
def update_subset_dropdown_options(n_clicks, matrix_data):
    """Updates the options for the subset dropdown based on the A-matrix."""
    if n_clicks is None or matrix_data is None:
        return []

    current_amatrix = pd.DataFrame(matrix_data)
    line_names = current_amatrix.index.tolist()
    options = [{'label': name, 'value': name} for name in line_names]
    return options

@app.callback(
    Output('subset-dropdown-dummy', 'options'),
    [Input('generate-amatrix-button-dummy', 'n_clicks')],
    [State('matrix-store-dummy', 'data')]
)
def update_subset_dropdown_options_dummy(n_clicks, matrix_data):
    """Updates the options for the subset dropdown based on the A-matrix for dummy progeny."""
    if n_clicks is None or matrix_data is None:
        return []

    current_amatrix = pd.DataFrame(matrix_data)
    line_names = current_amatrix.index.tolist()
    options = [{'label': name, 'value': name} for name in line_names]
    return options

@app.callback(
    Output('subset-matrix-display', 'children'),
    [Input('subset-dropdown', 'value')],
    [State('matrix-store', 'data')]
)
def display_subset_matrix(subset_values, matrix_data):
    """Displays the subset of the A-matrix."""
    if not subset_values or not matrix_data:
        return ""

    current_amatrix = pd.DataFrame(matrix_data)
    subset_amatrix = current_amatrix.loc[subset_values, subset_values]

    table_header = [html.Thead(html.Tr([html.Th('')] + [html.Th(col) for col in subset_amatrix.columns]))]
    table_body = [html.Tbody([
        html.Tr([html.Td(subset_amatrix.index[i])] + [html.Td(subset_amatrix.iloc[i, j]) for j in range(len(subset_amatrix.columns))]) for i in range(len(subset_amatrix))
    ])]

    return html.Table(table_header + table_body, className="table")

@app.callback(
    Output('subset-matrix-display-dummy', 'children'),
    [Input('subset-dropdown-dummy', 'value')],
    [State('matrix-store-dummy', 'data')]
)
def display_subset_matrix_dummy(subset_values, matrix_data):
    """Displays the subset of the A-matrix for dummy progeny."""
    if not subset_values or not matrix_data:
        return ""

    current_amatrix = pd.DataFrame(matrix_data)
    subset_amatrix = current_amatrix.loc[subset_values, subset_values]

    table_header = [html.Thead(html.Tr([html.Th('')] + [html.Th(col) for col in subset_amatrix.columns]))]
    table_body = [html.Tbody([
        html.Tr([html.Td(subset_amatrix.index[i])] + [html.Td(subset_amatrix.iloc[i, j]) for j in range(len(subset_amatrix.columns))]) for i in range(len(subset_amatrix))
    ])]

    return html.Table(table_header + table_body, className="table")

@app.server.route('/download')
def download_file():
    """Handles file download requests."""
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
    """Updates the download link for the subset matrix."""
    if not subset_values or current_amatrix is None:
        return dash.no_update

    subset_amatrix = current_amatrix.loc[subset_values, subset_values]

    temp_file_subset = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    subset_amatrix.to_csv(temp_file_subset)
    temp_file_subset.close()

    subset_matrix_link = f'/download?filename={temp_file_subset.name}&type=subset'

    return subset_matrix_link

@app.callback(
    Output('download-subset-link-dummy', 'href'),
    [Input('subset-dropdown-dummy', 'value')]
)
def update_subset_matrix_download_link_dummy(subset_values):
    """Updates the download link for the subset matrix for dummy progeny."""
    if not subset_values or current_amatrix is None:
        return dash.no_update

    subset_amatrix = current_amatrix.loc[subset_values, subset_values]

    temp_file_subset = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    subset_amatrix.to_csv(temp_file_subset)
    temp_file_subset.close()

    subset_matrix_link = f'/download?filename={temp_file_subset.name}&type=subset'

    return subset_matrix_link

@app.callback(
    Output('progeny-results', 'children'),
    [Input('find-progeny-button', 'n_clicks')],
    [State('progeny-line-dropdown', 'value'), State('progeny-line-dropdown-2', 'value')]
)
def find_progeny(n_clicks, line1, line2):
    """Finds and displays progeny for the specified lines."""
    if n_clicks is None or not line1 or not line2:
        return dash.no_update

    progeny_df = df[(df['MaleParent'] == line1) & (df['FemaleParent'] == line2) |
                    (df['MaleParent'] == line2) & (df['FemaleParent'] == line1)]

    if progeny_df.empty:
        return "No progeny found for the selected lines."

    return html.Ul([html.Li(f"{row['LineName']} (Female: {row['FemaleParent']}, Male: {row['MaleParent']})") for index, row in progeny_df.iterrows()])

@app.callback(
    Output('single-parent-progeny-results', 'children'),
    [Input('find-single-parent-progeny-button', 'n_clicks')],
    [State('single-parent-dropdown', 'value')]
)
def find_single_parent_progeny(n_clicks, parent):
    """Finds and displays progeny for the specified single parent."""
    if n_clicks is None or not parent:
        return dash.no_update

    progeny_df = df[(df['MaleParent'] == parent) | (df['FemaleParent'] == parent)]

    if progeny_df.empty:
        return "No progeny found for the selected parent."

    results = []
    for index, row in progeny_df.iterrows():
        if row['MaleParent'] == parent:
            male_parent = parent
            female_parent = row['FemaleParent']
        else:
            male_parent = row['MaleParent']
            female_parent = parent

        results.append(
            html.Li(
                f"{row['LineName']} (Female: {female_parent}, Male: {male_parent})"
            )
        )

    return html.Ul(results)

@app.callback(
    Output('combined-family-tree-image', 'children'),
    [Input('generate-combined-family-tree-button', 'n_clicks')],
    [State('combined-family-tree-dropdown-1', 'value'), State('combined-family-tree-dropdown-2', 'value')]
)
def generate_combined_family_tree(n_clicks, line1, line2):
    """Generates and displays the combined family tree for two selected lines."""
    if n_clicks is None or not line1 or line2 is None:
        return dash.no_update

    # Get ancestors and relationships for both lines
    ancestors1, relationships1 = find_ancestors(line1, filtered_df)
    ancestors2, relationships2 = find_ancestors(line2, filtered_df)

    all_lines = ancestors1.union(ancestors2, {line1, line2})
    all_relationships = relationships1 + relationships2

    # Determine the origin of each line (line1, line2, or shared)
    line_colors = {}
    for ancestor in ancestors1:
        line_colors[ancestor] = '#ADD8E6'  # Light blue for line1 ancestors
    for ancestor in ancestors2:
        if ancestor in line_colors:
            line_colors[ancestor] = '#FFFFE0'  # Light yellow for shared ancestors
        else:
            line_colors[ancestor] = '#FFB6C1'  # Light red for line2 ancestors

    # Include the selected lines
    line_colors[line1] = 'green'
    line_colors[line2] = 'green'

    # Merge the dataframes to include both sets of relatives
    relatives_df = filtered_df[filtered_df['LineName'].isin(all_lines)]

    # Create the graphviz digraph
    dot = graphviz.Digraph(comment='Combined Family Tree')
    dot.attr('node', shape='ellipse', style='filled')

    for _, row in relatives_df.iterrows():
        line_name = row['LineName']
        label = f"{line_name}"

        # Determine node color based on the origin
        node_color = line_colors.get(line_name, 'lightgrey')

        dot.node(line_name, label=label, fillcolor=node_color, fontcolor='black', color='black')

    # Track added edges to avoid duplication
    added_edges = set()

    for parent, child, role in all_relationships:
        if (parent, child) not in added_edges:
            if child in ancestors1 or child == line1:
                color = '#ADD8E6'  # Dark blue for line1
            elif child in ancestors2 or child == line2:
                color = '#FFB6C1'  # Dark red for line2
            else:
                color = 'black'  # Default black for shared
            dot.edge(parent, child, color=color)
            added_edges.add((parent, child))

    # Convert the graph to a PNG image
    tree_buffer = BytesIO()
    tree_buffer.write(dot.pipe(format='png'))
    tree_buffer.seek(0)
    encoded_tree = base64.b64encode(tree_buffer.read()).decode('utf-8')
    tree_src = f'data:image/png;base64,{encoded_tree}'

    return html.Img(src=tree_src, style=CUSTOM_CSS['image'])

@app.callback(
    Output('temp-progeny-tree-image', 'children'),
    [Input('generate-temp-progeny-tree-button', 'n_clicks')],
    [State('temp-female-parent-dropdown', 'value'), State('temp-male-parent-dropdown', 'value')]
)
def generate_temp_progeny_tree(n_clicks, female_parent, male_parent):
    """Generates and displays the family tree with a temporary progeny."""
    if n_clicks is None or not female_parent or not male_parent:
        return dash.no_update

    temp_progeny_name = "Temp_Progeny"
    temp_df = filtered_df.copy()
    temp_df = pd.concat([temp_df, pd.DataFrame([[temp_progeny_name, female_parent, male_parent]], columns=temp_df.columns)])

    # Find all ancestors
    all_ancestors, relationships = find_ancestors(temp_progeny_name, temp_df)
    all_lines = all_ancestors.union({temp_progeny_name})
    relatives_df = temp_df[temp_df['LineName'].isin(all_lines)]

    # Compute the A-matrix (akin to kinship matrix)
    sorted_relatives_df = sort_pedigree_df(relatives_df)
    kinship_matrix = compute_amatrix_diploid_revised(sorted_relatives_df)

    # Determine quantiles for kinship values
    kinship_values = kinship_matrix.loc[temp_progeny_name, all_lines].drop(temp_progeny_name)
    quantiles = kinship_values.quantile([0.25, 0.5, 0.75])

    def get_color(value, female_parent, male_parent):
        if value <= quantiles[0.25]:
            return '#ADD8E6'  # Dark blue if from one parent
        elif value <= quantiles[0.5]:
            return '#FFB6C1'  # Dark red if from the other parent
        else:
            return '#FFFFE0'  # Light yellow if from both

    dot = graphviz.Digraph(comment='Family Tree with Temporary Progeny')
    dot.attr('node', shape='ellipse', style='filled', fontsize='20')  # Increase font size

    for _, row in relatives_df.iterrows():
        line_name = row['LineName']
        label = f"{line_name}"

        # Determine node color based on kinship value
        if line_name != temp_progeny_name:
            kinship_value = kinship_matrix.loc[temp_progeny_name, line_name]
            label += f"\nKinship: {kinship_value:.2f}"
            color = get_color(kinship_value, female_parent, male_parent)
        else:
            kinship_value = kinship_matrix.loc[temp_progeny_name, temp_progeny_name]
            label += f"\nKinship: {kinship_value:.2f}"
            color = 'orange'  # Highlight the temporary progeny

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
    [State('dummy-progeny-name', 'value'),
     State('temp-female-parent-dropdown-main', 'value'),
     State('temp-male-parent-dropdown-main', 'value')]
)
def update_temp_progeny_list(add_clicks, reset_clicks, progeny_name, female_parent, male_parent):
    """Adds a dummy progeny to the list or resets the list."""
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

    list_items = [html.Li(f"{name} (Female: {female}, Male: {male})") for name, female, male in temp_progeny_list]
    return html.Ul(list_items)





def compute_amatrix_diploid_revised(pedigree_df):
    """Computes the A-matrix for a diploid species."""
    # Extract unique individuals
    individuals = pedigree_df['LineName'].unique()
    n = len(individuals)

    # Mapping from individual name to its index in the matrix
    index_map = {ind: i for i, ind in enumerate(individuals)}

    # Initialize A-matrix as an identity matrix
    A = np.identity(n)

    # Function to get index in A-matrix, return -1 if individual not present
    def get_index(ind):
        return index_map.get(ind, -1)

    for i, row in enumerate(pedigree_df.iterrows()):
        _, data = row
        ind_idx = get_index(data['LineName'])
        sire_idx = get_index(data['MaleParent'])
        dam_idx = get_index(data['FemaleParent'])

        if sire_idx == -1 and dam_idx == -1:  # No parents known
            A[ind_idx, ind_idx] = 1
            for j in range(n):
                if j != ind_idx:
                    A[ind_idx, j] = A[j, ind_idx] = 0

        elif sire_idx == -1:  # Only dam known
            A[ind_idx, ind_idx] = 1
            for j in range(n):
                if j != ind_idx:
                    A[ind_idx, j] = A[j, ind_idx] = 0.5 * A[dam_idx, j]

        elif dam_idx == -1:  # Only sire known
            A[ind_idx, ind_idx] = 1
            for j in range(n):
                if j != ind_idx:
                    A[ind_idx, j] = A[j, ind_idx] = 0.5 * A[sire_idx, j]

        else:  # Both parents known
            A[ind_idx, ind_idx] = 1 + 0.5 * A[sire_idx, dam_idx]
            for j in range(n):
                if j != ind_idx:
                    A[ind_idx, j] = A[j, ind_idx] = 0.5 * (A[sire_idx, j] + A[dam_idx, j])

    # Convert the matrix to a DataFrame with labeled indices
    labeled_A = pd.DataFrame(A, index=individuals, columns=individuals)

    return labeled_A

def sort_pedigree_df(pedigree_df):
    """Sorts the pedigree dataframe based on ancestry."""
    sorted_df = pd.DataFrame(columns=pedigree_df.columns)
    processed = set()

    def add_to_sorted_df(line_name):
        nonlocal sorted_df
        if line_name in processed:
            return
        row = pedigree_df[pedigree_df['LineName'] == line_name]
        if not row.empty:
            sire, dam = row['MaleParent'].values[0], row['FemaleParent'].values[0]
            if sire and sire not in processed:
                add_to_sorted_df(sire)
            if dam and dam not in processed:
                add_to_sorted_df(dam)
            sorted_df = pd.concat([sorted_df, row])
            processed.add(line_name)

    for line_name in pedigree_df['LineName']:
        add_to_sorted_df(line_name)

    return sorted_df

def old_find_ancestors(line_names, df, processed=None):
    """Finds ancestors for given line names."""
    if processed is None:
        processed = set()

    parents_df = df[df['LineName'].isin(line_names)][['MaleParent', 'FemaleParent']].dropna()
    parents = parents_df['MaleParent'].tolist() + parents_df['FemaleParent'].tolist()

    new_parents = [parent for parent in parents if parent not in processed]

    if not new_parents:
        return line_names

    processed.update(new_parents)
    all_relatives = list(set(line_names + new_parents))

    return old_find_ancestors(all_relatives, df, processed)

def old_find_descendants(line_names, df, processed=None):
    """Finds descendants for given line names."""
    if processed is None:
        processed = set()

    progeny_df = df[df['MaleParent'].isin(line_names) | df['FemaleParent'].isin(line_names)]
    progeny = progeny_df['LineName'].tolist()

    new_progeny = [child for child in progeny if child not in processed]

    if not new_progeny:
        return line_names

    processed.update(new_progeny)
    all_relatives = list(set(line_names + new_progeny))

    return old_find_descendants(all_relatives, df, processed)

def old_find_relatives(line_names, df):
    """Finds all relatives (ancestors and descendants) for given line names."""
    ancestors = old_find_ancestors(line_names, df)
    descendants = old_find_descendants(line_names, df)
    descendant_ancestors = [old_find_ancestors([desc], df) for desc in descendants]
    all_relatives = list(set(ancestors + descendants + [item for sublist in descendant_ancestors for item in sublist]))
    return all_relatives

@app.callback(Output('upload-status', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'), State('upload-data', 'last_modified')])
def upload_file(contents, filename, last_modified):
    """Handles file upload and updates the global dataframe."""
    if contents is None:
        return ""

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            uploaded_df = pd.read_csv(BytesIO(decoded))
        elif 'xls' in filename:
            # Assume that the user uploaded an Excel file
            uploaded_df = pd.read_excel(BytesIO(decoded))
        else:
            return "Unsupported file format"

        # Replace the global dataframe with the uploaded data
        global df, filtered_df, parents_set
        df = uploaded_df
        parents_set = set(df['MaleParent'].tolist() + df['FemaleParent'].tolist())
        mask = ~df['LineName'].isin(parents_set) & (df['MaleParent'].isna() | df['MaleParent'].isnull()) & (df['FemaleParent'].isna() | df['FemaleParent'].isnull())
        filtered_df = df[~mask]

        return f"Successfully uploaded {filename}"
    except Exception as e:
        print(e)
        return "There was an error processing the file"

def dummy_progeny_page_layout():
    """Layout for the dummy progeny page."""
    content = [
        dbc.Row(dbc.Col(html.H1("Dummy Progeny Page", className="app-header", style=CUSTOM_CSS['header']))),

        dbc.Row([
            dbc.Col(dcc.Input(
                id='dummy-progeny-name',
                type='text',
                placeholder='Enter Dummy Progeny Name',
                style={'width': '100%'}
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
            dbc.Col(html.Button("Add Dummy Progeny", id="add-temp-progeny-main-button", className="btn btn-warning", style=CUSTOM_CSS['button'])),
            dbc.Col(html.Button("Reset Dummy Progeny", id="reset-temp-progeny-main-button", className="btn btn-danger", style=CUSTOM_CSS['button']))
        ]),
        dbc.Row(dbc.Col(html.Div(id='temp-progeny-list-main'))),

        dbc.Row([
            dbc.Col(html.Button("Get Matrix", id="generate-amatrix-button-dummy", className="btn btn-info", style=CUSTOM_CSS['button']), width=3),
            dbc.Col(html.A("Download Full Matrix", id='download-full-link-dummy', href='', download='full_matrix.csv', className="btn btn-success", style=CUSTOM_CSS['button'])),
            dbc.Col(html.A("Download Subset Matrix", id='download-subset-link-dummy', href='', download='subset_matrix.csv', className="btn btn-warning", style=CUSTOM_CSS['button']))
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






if __name__ == '__main__':
    app.run_server(debug=True)
