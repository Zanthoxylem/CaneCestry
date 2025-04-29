import dash
from dash import html, dcc, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import flask
from flask import send_file
import numpy as np
import graphviz
from flask import Flask
from collections import defaultdict
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from numba import njit
import urllib.parse
import os
import time
import re
from dash.exceptions import PreventUpdate




#from dash.dependencies import Input, Output, State, ALL
#import dash
#import base64
#import pandas as pd
#from io import BytesIO
#import numba
#import uuid
#from dash import dash_table




# A threshold to skip clustering if matrix exceeds this size
MAX_CLUSTER_SIZE = 300000

# A compiled regex for names with digits + exactly one 'p' or 'P' and no other letters : For Polycrosses
pattern_poly_p = re.compile(r'^\d*[pP]\d*$')

# Create an output folder to hold matrix files
OUTPUT_DIR = os.path.join(os.getcwd(), "matrices")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Flask server
server = Flask(__name__)



#region Setting up data
# =============================================================================
# 1. Store Built-In (Default) Sugarcane Data and Create a Copy for df
# =============================================================================
default_df = pd.read_csv('/home/Zanthoxylum2117/Canecestry/Pedigree_25.txt', sep="\t")
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

CUSTOM_CSS = {
    'container': {'padding': '20px', 'margin-top': '20px', 'background-color': '#f9f9f9'},
    'header': {
        'textAlign': 'center',
        'padding': '10px',
        'color': '#2c3e50',
        'font-family': 'Arial',
        'font-weight': 'bold',
        'font-size': '50px'
    },
    'button': {
        'margin': '10px',
        'font-weight': 'bold',
        'background-color': '#2980b9',
        'color': 'white'
    },
    'table': {'overflowX': 'auto', 'margin-bottom': '20px', 'border': '1px solid #ccc'},
    'image': {'width': '100%', 'padding': '10px'},
    'dropdown': {'font-weight': 'bold', 'color': '#2980b9'},
    'static-button': {
        'position': 'absolute',
        'top': '10px',
        'right': '10px',
        'border-radius': '50%',
        'width': '85px',             # Increased from 75px
        'height': '85px',            # Increased from 75px
        'font-size': '45px',         # Increased from 40px
        'textAlign': 'center',
        'lineHeight': '65px',        # Adjusted to vertically center the icon/text
        'background-color': '#2980b9',
        'color': 'white',
        'border': 'none'
    },
    'home-button': {
        'top': '10px',
        'border-radius': '50%',
        'width': '85px',             # Increased from 75px
        'height': '85px',            # Increased from 75px
        'font-size': '45px',         # Increased from 40px
        'textAlign': 'center',
        'lineHeight': '65px',        # Adjusted line-height
        'background-color': '#2980b9',
        'color': 'white',
        'border': 'none'
    },
    'view-data-button': {
        'top': '10px',
        'border-radius': '50%',
        'width': '85px',             # Increased from 75px
        'height': '85px',            # Increased from 75px
        'font-size': '45px',         # Increased from 40px
        'textAlign': 'center',
        'lineHeight': '65px',        # Adjusted line-height
        'background-color': '#27ae60',
        'color': 'white',
        'border': 'none'
    },
    # Define new inline styles for the left and right button groups.
    'left_button_group_style' : {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '10px',
        'marginTop': '0px'         # Reduced top margin to move groups up
    },
    'right_button_group_style' : {
        'display': 'flex',
        'justifyContent': 'flex-end',
        'alignItems': 'center',
        'gap': '10px',
        'marginTop': '0px'         # Reduced top margin to move groups up
    }
}


#endregion

#region Layouts
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# =============================================================================
# 2. Layouts for Diffrent Pages
# =============================================================================
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# =============================================================================
# A. Layout for static items across entire site
# =============================================================================

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

def base_layout(content):
    """Generates the base layout with common structure."""
    return dbc.Container([
        # Header row with left and right button groups
        dbc.Row([
            dbc.Col(
                html.Div([
                    # Home button on the left
                    html.Button(
                        html.I(className="fas fa-home"),
                        id="home-button",
                        style={**CUSTOM_CSS['home-button'], 'position': 'static'}
                    ),
                    # View Data button, to the right of Home button
                    html.Button(
                        html.I(className="fas fa-table"),
                        id="view-data-button",
                        style={**CUSTOM_CSS['view-data-button'], 'position': 'static'}
                    )
                ], style={**CUSTOM_CSS['left_button_group_style']}),
                width=6
            ),
            dbc.Col(
                html.Div([
                    # Trash (Clear Data) button on the right
                    html.Button(
                        html.I(className="fas fa-trash"),
                        id="clear-data-button",
                        style={**CUSTOM_CSS['static-button'],
                               'background-color': '#e74c3c',
                               'position': 'static'}  # Remove absolute positioning
                    ),
                    # Help button ("?" icon) on the right
                    html.Button(
                        "?",
                        id="open-info-modal",
                        style={**CUSTOM_CSS['static-button'],
                               'position': 'static'}  # Remove absolute positioning
                    )
                ], style={**CUSTOM_CSS['right_button_group_style']}),
                width=6
            )
        ]),
        # Page title row
        dbc.Row(
            dbc.Col(
                html.H1(
                    "CaneCestry",
                    className="app-header",
                    style={**CUSTOM_CSS['header'], 'font-size': '60px'}
                )
            )
        ),
        # Content rows
        dbc.Row(
            dbc.Col(html.Div(id='common-content', children=content))
        ),
        dbc.Row(
            dbc.Col(html.Div(id='page-specific-content'))
        ),
        # Existing modals (Information & Data Viewer)
        dbc.Modal([
            dbc.ModalHeader("Information"),
            dbc.ModalBody(
                html.Div([
                    html.H4("Overview"),
                    html.P(
                        "CaneCestry 2.0 is a tool for managing and analyzing "
                        "pedigree data in sugarcane breeding programs. It offers "
                        "functionalities to visualize family trees, compute "
                        "kinship matrices, and explore progeny."
                    ),
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
        ], id="info-modal", size="lg", is_open=False),
    dbc.Modal([
        dbc.ModalHeader("Current Dataset"),
        dbc.ModalBody(
            html.Div([
                html.Div(id="current-data-table", style={'maxHeight': '400px', 'overflowY': 'scroll'}),
                html.Br(),
                html.Button("Make df", id="make-df-button", className="btn btn-primary"),
                html.Div(id="make-df-status")
            ])
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-data-modal", className="ml-auto")
        )
    ], id="data-modal", size="lg", is_open=False),
    # dcc.Store to hold the current dataset (including appended entries)
    dcc.Store(
        id="data-store",
        data=default_df.to_dict('records'),
        storage_type='local'
    ),

    ], fluid=True, style=CUSTOM_CSS['container'])



# =============================================================================
# B. Layout for Splash Page popup and buttons
# =============================================================================



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
                            "Upload a .csv (or .txt with tab separator) with columns: "
                            "LineName, MaleParent, FemaleParent",
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
                        html.Div(
                            id='splash-upload-status',
                            style={'marginTop': '10px', 'color': 'green'}
                        ),
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

        # 1) BUTTON FOR KINSHIP MATRIX PAGE
        dbc.Row(
            dbc.Col(
                html.A(
                    "Generate Kinship Matrix w/ Decendants",  # Renamed here
                    href="/main-page",
                    className="btn",
                    style={
                        "width": "300px",
                        "height": "80px",
                        "fontSize": "20px",
                        "marginBottom": "10px",
                        "backgroundColor": "#2980b9",
                        "color": "white",
                        "borderRadius": "10px",
                        "fontWeight": "bold",
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "textDecoration": "none"
                    }
                ),
                width="auto"
            ),
            justify="center"
        ),

        # 2) BUTTON FOR SUBSET PEDIGREE PAGE
        dbc.Row(
            dbc.Col(
                html.A(
                    "Generate Kinship Matrix w/o Decendants",  # New separate button
                    href="/subset-pedigree-page",
                    className="btn",
                    style={
                        "width": "300px",
                        "height": "80px",
                        "fontSize": "20px",
                        "marginBottom": "10px",
                        "backgroundColor": "#2980b9",
                        "color": "white",
                        "borderRadius": "10px",
                        "fontWeight": "bold",
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "textDecoration": "none"
                    }
                ),
                width="auto"
            ),
            justify="center"
        ),

        # REMAINING BUTTONS/ROWS...
        dbc.Row(
            dbc.Col(
                html.A(
                    "Pedigree Explorer",
                    href="/progeny-finder",
                    className="btn",
                    style={
                        "width": "300px",
                        "height": "80px",
                        "fontSize": "20px",
                        "marginBottom": "10px",
                        "backgroundColor": "#2980b9",
                        "color": "white",
                        "borderRadius": "10px",
                        "fontWeight": "bold",
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "textDecoration": "none"
                    }
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
                    style={
                        "width": "300px",
                        "height": "80px",
                        "fontSize": "20px",
                        "marginBottom": "10px",
                        "backgroundColor": "#2980b9",
                        "color": "white",
                        "borderRadius": "10px",
                        "fontWeight": "bold",
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "textDecoration": "none"
                    }
                ),
                width="auto"
            ),
            justify="center"
        )
    ]
    return base_layout(content)




# =============================================================================
# C. Layout for Kinship Matrix page that takes decendants into consideration
# =============================================================================

def main_page_layout():
    """Improved layout for the main page."""
    content = [
        # Row #1: Page title
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Generate Kinship Matrix",
                    className="app-header",
                    style=CUSTOM_CSS['header']
                )
            )
        ),

        # Row #2: Label for the slider
        dbc.Row([
            dbc.Col(
                html.Label("Select Calculation Method:", style={'font-weight': 'bold'}),
                width=12
            )
        ]),

        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dcc.Slider(
                                id='kinship-method-slider',
                                min=0,
                                max=1,
                                step=1,
                                marks={
                                    0: 'A Matrix (Henderson)',
                                    1: 'Coancestry Matrix (Henderson)'
                                },
                                value=1,
                                tooltip={"placement": "bottom", "always_visible": True},
                            )
                        ],
                        style={'width': '300px'}
                    ),
                    width="auto"
                )
            ],
            justify="center"  # <--- This will center the column horizontally
        ),




        # Row #4: Label + dropdown + input
        dbc.Row([
            dbc.Col(
                html.Label(
                    "Select or Paste Line Names (comma-separated):",
                    style={'font-weight': 'bold', 'margin-bottom': '10px'}
                ),
                width=12
            ),
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

        # Row #5: Display selected lines
        dbc.Row(
            dbc.Col(
                html.Ul(id="selected-line-names-list", children=[]),
                style={'margin-top': '10px'}
            )
        ),

        # Row #6: Buttons (Generate, Download, etc.)
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

        # Row #7: Loading/feedback area
        dbc.Row(
            dbc.Col(
                html.Div(
                    id='loading-output',
                    style={'margin-top': '20px'}
                )
            )
        ),

        # Row #8: Heatmap image
        dbc.Row(
            dbc.Col(
                html.Img(
                    id='heatmap-image',
                    src='',
                    style=CUSTOM_CSS['image']
                )
            )
        ),

        # Row #9: Subset selection
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

        # Hidden Divs, etc.
        html.Div(id='full-matrix-csv-path', style={'display': 'none'}),
        dcc.Store(id='matrix-store'),
    ]

    return base_layout(content)

# =============================================================================
# D. Layout for Page with Family tree and other features
# =============================================================================

def pedigree_explorer():
    """Layout for the progeny finder page."""
    content = [
        # Row #1: A dropdown for choosing progeny-module functions
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
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
                ),
                width=12
            )
        ]),

        # Row #2: A label for the slider
        dbc.Row([
            dbc.Col(
                html.Label("Select Calculation Method:", style={'font-weight': 'bold'}),
                width=12
            )
        ]),

        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dcc.Slider(
                                id='kinship-method-slider',
                                min=0,
                                max=1,
                                step=1,
                                marks={
                                    0: 'A Matrix (Henderson)',
                                    1: 'Coancestry Matrix (Henderson)'
                                },
                                value=1,
                                tooltip={"placement": "bottom", "always_visible": True},
                            )
                        ],
                        style={'width': '300px'}
                    ),
                    width="auto"
                )
            ],
            justify="center"  # <--- This will center the column horizontally
        ),

        # Row #4: A placeholder div for dynamic modules
        html.Div(id='selected-progeny-modules', children=[]),

        # Row #5: A single column with a "Back to Main Page" link
        dbc.Row([
            dbc.Col(
                html.A(
                    "Back to Main Page",
                    href="/main-page",
                    className="btn btn-secondary",
                    style=CUSTOM_CSS['button']
                ),
                width=12
            )
        ]),
    ]

    return base_layout(content)

# =============================================================================
# D. Layout for Page with Family tree and other features
# =============================================================================

def add_temp_entries():
    content = [
        dbc.Row(
            dbc.Col(
                html.H3("Add Pedigree Entries",
                        style=CUSTOM_CSS['header'])
            )
        ),

        # File upload area for a tab-delimited .txt
        dbc.Row(
            [
                dbc.Col(html.Label("Upload .txt pedigree entries:")),
                dbc.Col(
                    dcc.Upload(
                        id='add-ped-upload',
                        children=html.Div([
                            "Drag & Drop or ",
                            html.A("Select .txt File")
                        ]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center'
                        },
                        multiple=False
                    ),
                    width=6
                ),
                dbc.Col(
                    html.Div(id='add-ped-upload-status', style={'color': 'green'}),
                    width=12
                )
            ]
        ),

        html.Hr(),

        # Display rows with missing parents needing correction
        dbc.Row([
            dbc.Col(html.H4("Rows Needing Parent Corrections:"), width=12),
            dbc.Col(html.Div(id='add-ped-missing-parents-table'), width=12)
        ]),

        # Stores for invalid and valid rows
        dcc.Store(id='add-ped-missing-rows-store'),
        dcc.Store(id='add-ped-rows-store'),

        html.Hr(),

        # Button to navigate to subset calculation page
        dbc.Row([
            dbc.Col(
                html.A(
                    "Proceed to Subset Matrix Calculation",
                    href="/subset-pedigree-page",
                    className="btn btn-primary",
                    style=CUSTOM_CSS['button']
                ),
                width=12
            )
        ], style={'margin-top': '20px'}),

        html.Hr(),

        # Button to return to main page
        dbc.Row([
            dbc.Col(
                html.A(
                    "Back to Main Page",
                    href="/main-page",
                    className="btn btn-secondary",
                    style=CUSTOM_CSS['button']
                ),
                width=12
            )
        ]),
    ]
    return base_layout(content)






# =============================================================================
# D. Kinship matrix calculation without descendants
# =============================================================================
def subset_pedigree_page_layout():
    """
    A page layout where the user can pick lines of interest either by selecting
    from a dropdown or pasting a comma-separated list. The code then builds a
    reduced pedigree (lines + ancestors only) and computes the A-matrix for that subset.
    """
    content = [
        # Row #1: Page header
        dbc.Row(
            dbc.Col(
                html.H3(
                    "Subset Pedigree: Lines of Interest + Ancestors",
                    style=CUSTOM_CSS['header']
                )
            )
        ),

        # Row #2: Dropdown for lines of interest
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='subset-pedigree-lines-dropdown',
                    options=[{'label': ln, 'value': ln} for ln in df['LineName'].unique()],
                    multi=True,
                    placeholder="Select lines of interest...",
                    style=CUSTOM_CSS['dropdown']
                ),
                width=12
            )
        ]),

        # Row #3: Input for comma-separated lines
        dbc.Row([
            dbc.Col(
                dcc.Input(
                    id='subset-pedigree-paste-input',
                    type='text',
                    placeholder="Paste comma-separated line names here...",
                    style={'margin-top': '10px', 'width': '100%'}
                ),
                width=12
            )
        ]),

        # Row #4: Label for slider
        dbc.Row([
            dbc.Col(
                html.Label("Select Calculation Method:", style={'font-weight': 'bold'}),
                width=12
            )
        ]),

        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dcc.Slider(
                                id='kinship-method-slider',
                                min=0,
                                max=1,
                                step=1,
                                marks={
                                    0: 'A Matrix (Henderson)',
                                    1: 'Coancestry Matrix (Henderson)'
                                },
                                value=1,
                                tooltip={"placement": "bottom", "always_visible": True},
                            )
                        ],
                        style={'width': '300px'}
                    ),
                    width="auto"
                )
            ],
            justify="center"  # <--- This will center the column horizontally
        ),

        # Row #6: "Build Subset & Compute Matrix" button
        dbc.Row([
            dbc.Col(
                html.Button(
                    "Build Subset & Compute Matrix",
                    id="subset-build-matrix-button",
                    className="btn btn-info",
                    style=CUSTOM_CSS['button']
                ),
                width=12
            )
        ]),

        # Row #7: Output area for subset-pedigree-output
        dbc.Row([
            dbc.Col(
                html.Div(
                    id='subset-pedigree-output',
                    style={'marginTop': '20px'}
                ),
                width=12
            )
        ]),

        html.Hr(),

        # Row #8: Subset the computed matrix label
        dbc.Row([
            dbc.Col(
                html.H4("Subset the Computed Matrix"),
                width=12
            )
        ]),

        html.Div(id='subset-pedigree-full-matrix-csv-path', style={'display': 'none'}),
        dcc.Store(id='subset-pedigree-matrix-store'),

        # Row #9: Sub-subsetting the matrix
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='subset-pedigree-subset-dropdown',
                    options=[],
                    multi=True,
                    placeholder="Select lines for subset matrix...",
                    style=CUSTOM_CSS['dropdown']
                ),
                width=9
            ),
            dbc.Col(
                html.A(
                    "Download Subset Matrix",
                    id='subset-pedigree-download-subset-link',
                    href='',
                    download='subset_matrix.csv',
                    className="btn btn-warning",
                    style=CUSTOM_CSS['button']
                ),
                width=3
            )
        ], style={'margin-top': '20px'}),

        # Row #10: Pasting lines for sub-subset
        dbc.Row([
            dbc.Col(
                dcc.Input(
                    id='subset-pedigree-subset-paste-input',
                    type='text',
                    placeholder="Paste comma-separated line names here...",
                    style={'width': '100%'}
                ),
                width=12
            )
        ], style={'margin-top': '20px'}),

        # Row #11: Display sub-subset table
        dbc.Row([
            dbc.Col(
                html.Div(id='subset-pedigree-subset-display'),
                width=12
            )
        ]),

        # Row #12: "Back to Main Page"
        dbc.Row([
            dbc.Col(
                html.A(
                    "Back to Main Page",
                    href="/main-page",
                    className="btn btn-secondary",
                    style=CUSTOM_CSS['button']
                ),
                width=12
            )
        ]),
    ]

    return base_layout(content)

#endregion




#region Return Selected_Lines


@app.callback(
    Output('subset-pedigree-lines-dropdown', 'value'),
    [Input('subset-pedigree-lines-dropdown', 'value'),
     Input('subset-pedigree-paste-input', 'value')]
)
def update_subset_line_selection(selected_lines, pasted_lines):
    if pasted_lines:
        pasted_list = [name.strip() for name in pasted_lines.split(',')]
        all_lines = df['LineName'].unique()
        valid_lines = [line for line in pasted_list if line in all_lines]
        if selected_lines:
            valid_lines = list(set(selected_lines + valid_lines))
        return valid_lines
    return selected_lines

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@app.callback(
    Output('line-name-dropdown', 'value'),
    [Input('line-name-dropdown', 'value'),
     Input('paste-line-names', 'value')]
)
def update_line_selection(selected_lines, pasted_lines):
    if pasted_lines:
        pasted_list = [name.strip() for name in pasted_lines.split(',')]
        all_lines = df['LineName'].unique()
        valid_lines = [line for line in pasted_list if line in all_lines]
        if selected_lines:
            valid_lines = list(set(selected_lines + valid_lines))
        return valid_lines
    return selected_lines

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@app.callback(
    Output('subset-pedigree-subset-dropdown', 'value'),
    [
        Input('subset-pedigree-subset-dropdown', 'value'),
        Input('subset-pedigree-subset-paste-input', 'value')
    ],
    [State('subset-pedigree-matrix-store', 'data')]
)
def update_subset_pedigree_sub_subset(selected_lines, pasted_lines, matrix_data):
    """
    Merge pasted line names into the dropdown's current selection.
    Only keep lines that actually exist in the matrix's index.
    """
    # If we haven't computed/loaded the matrix yet, do nothing
    if not matrix_data:
        return selected_lines or []

    # Convert the stored dictionary back to a DataFrame to get valid line names
    current_amatrix = pd.DataFrame(matrix_data)
    valid_line_names = current_amatrix.index.tolist()

    # Start from whichever lines are currently selected in the dropdown
    selected_lines = selected_lines or []

    if pasted_lines:
        # Split the pasted text on commas
        pasted_list = [name.strip() for name in pasted_lines.split(',')]
        # Keep only those which exist in the matrix
        valid_pasted = [ln for ln in pasted_list if ln in valid_line_names]
        # Merge them with the already selected lines
        selected_lines = list(set(selected_lines + valid_pasted))

    return selected_lines


#endregion

#region Subset pedigree matrix logic
def build_subset_pedigree(lines_of_interest, full_pedigree_df):
    """
    Given a list of target lines, returns a subset DataFrame containing
    only those lines + all of their ancestors (paternal and maternal).
    Descendants (i.e., progeny) that are irrelevant to these lines
    are excluded, preventing huge expansions of the pedigree.
    """
    relevant_lines = set(lines_of_interest)

    def collect_ancestors(line_name):
        sub = full_pedigree_df[full_pedigree_df["LineName"] == line_name]
        if sub.empty:
            return
        male_parent = sub.iloc[0]["MaleParent"]
        female_parent = sub.iloc[0]["FemaleParent"]

        if pd.notna(male_parent) and male_parent not in relevant_lines:
            relevant_lines.add(male_parent)
            collect_ancestors(male_parent)

        if pd.notna(female_parent) and female_parent not in relevant_lines:
            relevant_lines.add(female_parent)
            collect_ancestors(female_parent)

    for line in lines_of_interest:
        collect_ancestors(line)

    subset_df = full_pedigree_df[full_pedigree_df["LineName"].isin(relevant_lines)].copy()
    return subset_df


@app.callback(
    [
        Output('subset-pedigree-output', 'children'),
        Output('subset-pedigree-matrix-store', 'data'),
        Output('subset-pedigree-full-matrix-csv-path', 'children')
    ],
    Input('subset-build-matrix-button', 'n_clicks'),
    State('subset-pedigree-lines-dropdown', 'value'),
    State('kinship-method-slider', 'value')  # <--- add slider State
)
def compute_subset_matrix(n_clicks, selected_lines, method_choice):
    if not n_clicks or not selected_lines:
        return dash.no_update, dash.no_update, dash.no_update

    subset_df = build_subset_pedigree(selected_lines, df)

        # Decide which function to call:
    if method_choice == 0:
        A_sub = compute_amatrix_diploid(subset_df)
    elif method_choice == 1:
        A_sub = compute_amatrix_polyploid(subset_df)



    n_lines = len(A_sub)
    if n_lines == 0:
        return "No lines found in subset.", {}, ""

    # Plot either a clustermap or simple heatmap
    if n_lines <= MAX_CLUSTER_SIZE:
        heatmap_plot = sns.clustermap(
            A_sub,
            method='average',
            cmap='Spectral',
            figsize=(12, 8),
            row_cluster=True,
            col_cluster=True
        )
        plt.close(heatmap_plot.fig)
        heatmap_buf = BytesIO()
        heatmap_plot.savefig(heatmap_buf, format='png')
        heatmap_buf.seek(0)
        encoded_heatmap = base64.b64encode(heatmap_buf.read()).decode('utf-8')
        heatmap_src = f"data:image/png;base64,{encoded_heatmap}"
    else:
        plt.figure(figsize=(12, 8))
        sns.heatmap(A_sub, cmap='Spectral')
        plt.title(f"Heatmap (No Clustering) for {n_lines} lines > {MAX_CLUSTER_SIZE}")
        heatmap_buf = BytesIO()
        plt.savefig(heatmap_buf, format='png')
        heatmap_buf.seek(0)
        plt.close()
        encoded_heatmap = base64.b64encode(heatmap_buf.read()).decode('utf-8')
        heatmap_src = f"data:image/png;base64,{encoded_heatmap}"

    # Save CSV
    file_name = f"subset_pedigree_matrix_{int(time.time())}.csv"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    A_sub.to_csv(file_path)
    safe_filename_subset = urllib.parse.quote(file_path)
    # We'll store it in a hidden div so that we can use it for subset downloading
    download_link = f'/download?filename={safe_filename_subset}&type=subset'

    # Also store the matrix in a dictionary so we can re-subset it
    matrix_dict = A_sub.to_dict()

    return (
        html.Div([
            html.Img(
                src=heatmap_src,
                style={'width': '100%', 'maxHeight': '800px', 'overflow': 'auto'}
            ),
            html.Hr(),
            html.A(
                "Download Subset Matrix CSV",
                href=download_link,
                className="btn btn-warning",
                style=CUSTOM_CSS['button']
            )
        ]),
        matrix_dict,  # for subset usage
        file_path     # store the file path for potential direct referencing
    )


# <<< ADDED: Now replicate the logic for subsetting that matrix >>>
@app.callback(
    Output('subset-pedigree-subset-dropdown', 'options'),
    Input('subset-pedigree-matrix-store', 'data')
)
def update_subset_pedigree_dropdown_options(matrix_data):
    if not matrix_data:
        return []
    current_amatrix = pd.DataFrame(matrix_data)
    line_names = current_amatrix.index.tolist()
    return [{'label': ln, 'value': ln} for ln in line_names]


@app.callback(
    Output('subset-pedigree-subset-display', 'children'),
    Input('subset-pedigree-subset-dropdown', 'value'),
    State('subset-pedigree-matrix-store', 'data')
)
def display_subset_pedigree_matrix(subset_values, matrix_data):
    if not subset_values or not matrix_data:
        return ""
    current_amatrix = pd.DataFrame(matrix_data)
    subset_amatrix = current_amatrix.loc[subset_values, subset_values]

    table_header = [
        html.Thead(
            html.Tr([html.Th('')] + [html.Th(col) for col in subset_amatrix.columns])
        )
    ]
    table_body = [
        html.Tbody([
            html.Tr([html.Td(subset_amatrix.index[i])] +
                    [html.Td(subset_amatrix.iloc[i, j]) for j in range(len(subset_amatrix.columns))])
            for i in range(len(subset_amatrix))
        ])
    ]
    return html.Table(table_header + table_body, className="table")


@app.callback(
    Output('subset-pedigree-download-subset-link', 'href'),
    Input('subset-pedigree-subset-dropdown', 'value'),
    State('subset-pedigree-matrix-store', 'data')
)
def update_subset_pedigree_matrix_download_link(subset_values, matrix_data):
    if not subset_values or not matrix_data:
        return dash.no_update
    current_amatrix = pd.DataFrame(matrix_data)
    subset_amatrix = current_amatrix.loc[subset_values, subset_values]
    file_name = f"subset_pedigree_resubset_{int(time.time())}.csv"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    subset_amatrix.to_csv(file_path)
    safe_filename_subset = urllib.parse.quote(file_path)
    return f'/download?filename={safe_filename_subset}&type=subset'
# <<< END ADDED >>>






#endregion

#region Logic behind webpage
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    """Displays the appropriate page based on the URL path."""
    if pathname == '/main-page':
        return main_page_layout()
    elif pathname == '/progeny-finder':
        return pedigree_explorer()
    elif pathname == '/dummy-progeny-matrix':
        return add_temp_entries()
    elif pathname == '/subset-pedigree-page':
        return subset_pedigree_page_layout()
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


#endregion

#region Logic behind Updating the view data modal


@app.callback(
    Output('data-modal', 'is_open'),
    [Input('view-data-button', 'n_clicks'),
     Input('close-data-modal', 'n_clicks')],
    [State('data-modal', 'is_open')]
)
def toggle_data_modal(view_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    return not is_open





@app.callback(
    Output('current-data-table', 'children'),
    Input('data-store', 'data')
)
def update_current_data_table(data):
    if not data:
        return "No data available."

    # Create table headers from keys of the first record
    headers = list(data[0].keys())
    table_header = html.Thead(html.Tr([html.Th(col) for col in headers]))

    # Create table rows for each record
    table_body = html.Tbody([
        html.Tr([html.Td(record.get(col, "")) for col in headers]) for record in data
    ])

    return html.Table([table_header, table_body], className="table")


@app.callback(
    Output('data-store', 'data'),
    Input('clear-data-button', 'n_clicks'),  # or any callback that modifies your dataset
    prevent_initial_call=True
)
def update_data_store(n_clicks):
    # When clearing data, reset to the original sugarcane dataset.
    global df, default_df
    if n_clicks:
        df = default_df.copy()
    return df.to_dict('records')

#endregion




#region Logic behind adding data and the web elements associated

# =============================================================================
# 3. Callback to Clear Data (reset df to default_df)
# =============================================================================
@app.callback(
    [Output('clear-data-button', 'children'),
     Output('data-store', 'data', allow_duplicate=True)],
    Input('clear-data-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_uploaded_data(n_clicks):
    global df, default_df, filtered_df, parents_set
    if n_clicks:
        df = default_df.copy()
        parents_set = set(df['MaleParent'].tolist() + df['FemaleParent'].tolist())
        mask = (
            ~df['LineName'].isin(parents_set)
            & (df['MaleParent'].isna() | df['MaleParent'].isnull())
            & (df['FemaleParent'].isna() | df['FemaleParent'].isnull())
        )
        filtered_df = df[~mask]
        return [html.I(className="fas fa-trash"), " Cleared!"], default_df.to_dict('records')
    return dash.no_update, dash.no_update



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


@app.callback(
    Output('make-df-status', 'children'),
    Input('make-df-button', 'n_clicks'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def make_df_from_store(n_clicks, store_data):
    global df, filtered_df, parents_set
    if n_clicks:
        # Convert stored data to DataFrame
        new_df = pd.DataFrame(store_data)
        df = new_df.copy()
        # Recalculate the filtered dataframe based on updated df
        parents_set = set(df['MaleParent'].tolist() + df['FemaleParent'].tolist())
        mask = (
            ~df['LineName'].isin(parents_set)
            & (df['MaleParent'].isna() | df['MaleParent'].isnull())
            & (df['FemaleParent'].isna() | df['FemaleParent'].isnull())
        )
        filtered_df = df[~mask]
        return "Global df updated from store."
    return ""





#endregion

#region Find Ancestors/Descendants/Relatives

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

    all_relatives = list(all_ancestors.union(all_descendants).union(line_names))
    return all_relatives, all_relationships, generations

#endregion

#region Maternal and Paternal Line logic

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

#endregion

#region Matrix Math and Sorting

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

            if pd.notna(sire) and sire in pedigree_df['LineName'].values:
                add_to_sorted_df(sire)
            if pd.notna(dam) and dam in pedigree_df['LineName'].values:
                add_to_sorted_df(dam)

            nonlocal sorted_df
            sorted_df = pd.concat([sorted_df, row], ignore_index=True)
            processed.add(line_name)

    for ln in pedigree_df['LineName']:
        add_to_sorted_df(ln)

    return sorted_df


@njit
def _build_matrix_numba(n, sire_idxs, dam_idxs):
    """
    Core Henderson logic for diploid species.
    """
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        s_i = sire_idxs[i]
        d_i = dam_idxs[i]
        if s_i == -1 and d_i == -1:
            A[i, i] = 1.0
        elif s_i != -1 and d_i == -1:
            A[i, i] = 1.0
            A[i, :i] = 0.5 * A[s_i, :i]
            A[:i, i] = A[i, :i]
        elif s_i == -1 and d_i != -1:
            A[i, i] = 1.0
            A[i, :i] = 0.5 * A[d_i, :i]
            A[:i, i] = A[i, :i]
        else:
            A[i, i] = 1.0 + 0.5 * A[s_i, d_i]
            temp = 0.5 * (A[s_i, :i] + A[d_i, :i])
            A[i, :i] = temp
            A[:i, i] = temp
    return A

def compute_amatrix_diploid(pedigree_df):
    """
    Builds the additive relationship matrix (A-matrix) using Henderson’s Method (Diploid),
    treating 'unknown' (case-insensitive) exactly like missing parents.
    """
    # Sort the pedigree so parents appear before offspring
    sorted_df = sort_pedigree_df(pedigree_df)

    # List of individuals in sorted order
    individuals = sorted_df['LineName'].tolist()

    # Map each individual to its matrix index
    idx_map = {ind: i for i, ind in enumerate(individuals)}
    n = len(individuals)

    # Arrays to hold the sire/dam indices for each individual
    sire_idxs = np.full(n, -1, dtype=np.int64)
    dam_idxs = np.full(n, -1, dtype=np.int64)

    for i in range(n):
        row = sorted_df.iloc[i]
        sire = row['MaleParent']
        dam = row['FemaleParent']

        # Treat NaN or 'unknown'/'Unknown' as no parent
        if pd.notna(sire) and str(sire).lower() != 'unknown' and sire in idx_map:
            sire_idxs[i] = idx_map[sire]

        if pd.notna(dam) and str(dam).lower() != 'unknown' and dam in idx_map:
            dam_idxs[i] = idx_map[dam]

    # Build the relationship matrix via Numba-accelerated Henderson logic
    A = _build_matrix_numba(n, sire_idxs, dam_idxs)
    return pd.DataFrame(A, index=individuals, columns=individuals)


def compute_amatrix_polyploid(pedigree_df):

    sorted_df = sort_pedigree_df(pedigree_df)
    individuals = sorted_df['LineName'].tolist()



    """
    Same Henderson logic as diploid, but all values are divided by 2.
    """
    A_dip = compute_amatrix_diploid(pedigree_df)
    A_poly = A_dip / 2.0


    return pd.DataFrame(A_poly, index=individuals, columns=individuals)

#endregion

#region Descendant Tree
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
    for gen in range(max(generations.keys()) + 1):
        subset_descendants.update(generations[gen])

    subset_relationships = []
    for parent, child, role in relationships:
        if parent in subset_descendants and child in subset_descendants:
            subset_relationships.append((parent, child, role))

    dot = graphviz.Digraph(comment='Descendant Tree')
    dot.attr('node', shape='ellipse', style='filled')
    for descendant in subset_descendants:
        label = f"{descendant}"
        node_color = 'lightgrey'
        if descendant == selected_line_name:
            node_color = 'green'
        dot.node(descendant, label=label, fillcolor=node_color, fontcolor='black', color='black')

    for parent, child, role in subset_relationships:
        dot.edge(parent, child, color='black')

    tree_buffer = BytesIO()
    tree_buffer.write(dot.pipe(format='png'))
    tree_buffer.seek(0)
    encoded_tree = base64.b64encode(tree_buffer.read()).decode('utf-8')
    tree_src = f'data:image/png;base64,{encoded_tree}'
    return html.Img(src=tree_src, style=CUSTOM_CSS['image'])

#endregion

#region Family tree generation and generation depth slider

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
        State('generation-depth-slider', 'value'),
        State('kinship-method-slider', 'value')
    ]
)
def generate_family_tree(n_clicks, lineage_mode, selected_line_name, generation_depth, method_choice):
    if not n_clicks or not selected_line_name:
        return dash.no_update

    # Compute the full family tree for the selected line
    all_ancestors, relationships, generations = find_ancestors(selected_line_name, filtered_df)
    full_nodes = all_ancestors.union({selected_line_name})

    # Compute the full kinship matrix using all relatives from the complete tree
    full_relatives_df = filtered_df[filtered_df['LineName'].isin(full_nodes)]
    sorted_full_relatives_df = sort_pedigree_df(full_relatives_df)
    if method_choice == 0:
        full_amatrix = compute_amatrix_diploid(sorted_full_relatives_df)
    elif method_choice == 1:
        full_amatrix = compute_amatrix_polyploid(sorted_full_relatives_df)

    # Compute the visual subset based on slider (generation depth)
    subset_nodes = set()
    for gen in range(generation_depth + 1):
        subset_nodes.update(generations.get(gen, []))
    subset_nodes.add(selected_line_name)  # Ensure the selected line is included

    # Use the full kinship matrix for coloring: only get values for nodes that are in the visual subset.
    kinship_values = {line: full_amatrix.loc[selected_line_name, line]
                      for line in full_amatrix.index if line in subset_nodes}

    # Also, restrict relationships to nodes in the visual subset
    subset_relationships = [(p, c, role) for p, c, role in relationships
                            if p in subset_nodes and c in subset_nodes]

    # Classification for node styling remains the same
    founders = set()
    poly_founders = set()
    half_defined = set()
    fully_defined = set()
    parent_count = {line: 0 for line in subset_nodes}
    for parent, child, _ in subset_relationships:
        if child in parent_count:
            parent_count[child] += 1

    for line in subset_nodes:
        num_parents = parent_count.get(line, 0)
        if num_parents == 0:
            founders.add(line)
            if "POLY" in line.upper() or pattern_poly_p.match(line):
                poly_founders.add(line)
        elif num_parents == 1:
            half_defined.add(line)
        else:
            fully_defined.add(line)

    # Set up the color mapping for kinship values
    kin_vals = list(kinship_values.values())
    norm = Normalize(vmin=min(kin_vals), vmax=max(kin_vals))
    cmap = cm.get_cmap('Spectral')

    def get_node_color(kval):
        color = cmap(norm(kval))
        return '#{:02x}{:02x}{:02x}'.format(
            int(color[0]*255),
            int(color[1]*255),
            int(color[2]*255)
        )

    # Here we compute the maternal and paternal lines for the selected node.
    maternal_line = set(get_maternal_line(selected_line_name, filtered_df))
    paternal_line = set(get_paternal_line(selected_line_name, filtered_df))

    dot = graphviz.Digraph(comment='Family Tree')
    dot.attr('node', shape='ellipse', style='filled', fontsize='20')

    # Draw nodes with potential highlighting
    for line in subset_nodes:
        kv = kinship_values.get(line, 0.0)
        label = f"{line}\nKinship: {kv:.2f}"
        node_color = get_node_color(kv)

        # Determine default node parameters
        node_style = None  # leave default style
        node_penwidth = "1"  # default pen width

        # Adjust based on the radio button value (lineage_mode)
        if lineage_mode == 'maternal' and line in maternal_line:
            node_penwidth = "3"
            node_style = "bold"
        elif lineage_mode == 'paternal' and line in paternal_line:
            node_penwidth = "3"
            node_style = "bold"
        elif lineage_mode == 'both' and (line in maternal_line or line in paternal_line):
            node_penwidth = "3"
            node_style = "bold"
        # else: if lineage_mode is 'none', do nothing extra

        # Incorporate additional styling based on your original classifications:
        if line in poly_founders:
            dot.node(line, label=label, fillcolor=node_color, fontcolor='black',
                     color='black', style=node_style if node_style else "dashed", penwidth=node_penwidth if node_penwidth else "3")
        elif line in founders:
            dot.node(line, label=label, fillcolor=node_color, fontcolor='black',
                     color='black', penwidth=node_penwidth if node_penwidth else "3")
        elif line in half_defined:
            dot.node(line, label=label, fillcolor=node_color, fontcolor='black',
                     color='black', penwidth="2", style="dotted")
        else:
            dot.node(line, label=label, fillcolor=node_color, fontcolor='black', color='black')

    # Draw edges as before
    for parent, child, role in subset_relationships:
        edge_color = 'blue' if role == 'male' else ('red' if role == 'female' else 'black')
        dot.edge(parent, child, color=edge_color)

    pedigree_summary = f"""Pedigree Completeness:
    Founders: {len(founders)}
    Poly Crosses: {len(poly_founders)}
    Half-Defined Genotypes: {len(half_defined)}
    Fully Defined Genotypes: {len(fully_defined)}
    """
    dot.node(
        "pedigree_summary",
        label=pedigree_summary,
        shape="box",
        fontsize="30",
        style="filled",
        fillcolor="white",
        fontcolor="black",
        width="1",
        height="1",
    )
    dot.edge("pedigree_summary", selected_line_name, style="invis")

    # Generate and encode the image
    tree_buffer = BytesIO()
    tree_buffer.write(dot.pipe(format='png'))
    tree_buffer.seek(0)
    encoded_tree = base64.b64encode(tree_buffer.read()).decode('utf-8')
    tree_src = f'data:image/png;base64,{encoded_tree}'
    return html.Img(src=tree_src, style=CUSTOM_CSS['image'])




#endregion



#region Old find Ancestors, Descendants, Relatives

@app.callback(
    Output('selected-line-names-list', 'children'),
    [Input('line-name-dropdown', 'value')]
)
def update_selected_line_names_list(selected_line_names):
    if not selected_line_names:
        return "No lines selected."
    return [html.Li(name) for name in selected_line_names]


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
    progeny_df = df[df['MaleParent'].isin(line_names) | df['FemaleParent'].isin(line_names)]
    progeny = progeny_df['LineName'].tolist()
    new_progeny = [child for child in progeny if child not in processed]

    if not new_progeny:
        return line_names

    processed.update(new_progeny)
    all_relatives = list(set(line_names + new_progeny))
    return old_find_descendants(all_relatives, df, processed)


def old_find_relatives(line_names, df):
    ancestors = old_find_ancestors(line_names, df)
    descendants = old_find_descendants(line_names, df)
    descendant_ancestors = [old_find_ancestors([d], df) for d in descendants]
    flat_desc_ances = [item for sublist in descendant_ancestors for item in sublist]
    all_relatives = list(set(ancestors + descendants + flat_desc_ances))
    return all_relatives


#endregion


@app.callback(
    [
        Output('heatmap-image', 'src'),
        Output('download-full-link', 'href'),
        Output('full-matrix-csv-path', 'children'),
        Output('matrix-store', 'data')
    ],
    [Input('generate-amatrix-button', 'n_clicks')],
    [State('line-name-dropdown', 'value'),
     State('kinship-method-slider', 'value')]
)
def generate_amatrix_and_heatmap(n_clicks, selected_line_names, method_choice):
    if n_clicks is None or not selected_line_names:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    all_related_lines = old_find_relatives(selected_line_names, filtered_df)
    relatives_df = filtered_df[filtered_df['LineName'].isin(all_related_lines)]

    # Select method based on slider value
    if method_choice == 0:
        current_amatrix = compute_amatrix_diploid(relatives_df)  # Standard Henderson
    elif method_choice == 1:
        current_amatrix = compute_amatrix_polyploid(relatives_df)  # Modified Henderson for Polyploids


    file_name = f"full_matrix_{int(time.time())}.csv"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    current_amatrix.to_csv(file_path)
    safe_filename_full = urllib.parse.quote(file_path)
    full_matrix_link = f'/download?filename={safe_filename_full}&type=full'

    n_lines = len(current_amatrix)
    if n_lines == 0:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if n_lines <= MAX_CLUSTER_SIZE:
        heatmap_plot = sns.clustermap(current_amatrix, method='average', cmap='Spectral', figsize=(15, 15))
        plt.close(heatmap_plot.fig)
        heatmap_buffer = BytesIO()
        heatmap_plot.savefig(heatmap_buffer, format='png')
        heatmap_buffer.seek(0)
        encoded_heatmap = base64.b64encode(heatmap_buffer.read()).decode('utf-8')
        heatmap_src = f'data:image/png;base64,{encoded_heatmap}'
    else:
        plt.figure(figsize=(15, 15))
        sns.heatmap(current_amatrix, cmap='Spectral')
        heatmap_buffer = BytesIO()
        plt.savefig(heatmap_buffer, format='png')
        heatmap_buffer.seek(0)
        plt.close()
        encoded_heatmap = base64.b64encode(heatmap_buffer.read()).decode('utf-8')
        heatmap_src = f'data:image/png;base64,{encoded_heatmap}'

    store_data = current_amatrix.to_dict()
    return heatmap_src, full_matrix_link, file_path, store_data



@app.callback(
    [
        Output('heatmap-image-dummy', 'src'),
        Output('download-full-link-dummy', 'href'),
        Output('full-matrix-csv-path-dummy', 'children'),
        Output('matrix-store-dummy', 'data')
    ],
    [Input('generate-amatrix-button-dummy', 'n_clicks')],
    [
        State('temp-progeny-list-main', 'children'),
        State('kinship-method-slider', 'value')  # <--- add slider state
    ]
)
def generate_amatrix_and_heatmap_dummy(n_clicks, temp_progeny_list_children, method_choice):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    temp_df = filtered_df.copy()
    for dummy_progeny in temp_progeny_list:
        temp_df = pd.concat([temp_df, pd.DataFrame([dummy_progeny], columns=temp_df.columns)])

    progeny_names = [p[0] for p in temp_progeny_list]
    all_related_lines = old_find_relatives(progeny_names, temp_df)
    relatives_df = temp_df[temp_df['LineName'].isin(all_related_lines)]

    # Switch based on the slider
    if method_choice == 0:
        current_amatrix = compute_amatrix_diploid(relatives_df)
    elif method_choice == 1:
        current_amatrix = compute_amatrix_polyploid(relatives_df)


    file_name = f"dummy_full_matrix_{int(time.time())}.csv"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    current_amatrix.to_csv(file_path)
    safe_filename_full = urllib.parse.quote(file_path)
    full_matrix_link = f'/download?filename={safe_filename_full}&type=full'

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

    store_data = current_amatrix.to_dict()
    return heatmap_src, full_matrix_link, file_path, store_data


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

    table_header = [
        html.Thead(
            html.Tr([html.Th('')] + [html.Th(col) for col in subset_amatrix.columns])
        )
    ]
    table_body = [
        html.Tbody([
            html.Tr([html.Td(subset_amatrix.index[i])] +
                    [html.Td(subset_amatrix.iloc[i, j]) for j in range(len(subset_amatrix.columns))])
            for i in range(len(subset_amatrix))
        ])
    ]
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

    table_header = [
        html.Thead(
            html.Tr([html.Th('')] + [html.Th(col) for col in subset_amatrix.columns])
        )
    ]
    table_body = [
        html.Tbody([
            html.Tr([html.Td(subset_amatrix.index[i])] +
                    [html.Td(subset_amatrix.iloc[i, j]) for j in range(len(subset_amatrix.columns))])
            for i in range(len(subset_amatrix))
        ])
    ]
    return html.Table(table_header + table_body, className="table")


@app.server.route('/download')
def download_file():
    """Handles file download requests from the generated CSV output files."""
    filename_quoted = flask.request.args.get('filename')
    matrix_type = flask.request.args.get('type', 'full')
    if not filename_quoted:
        return "File not found", 404
    filename = urllib.parse.unquote(filename_quoted)
    return send_file(
        filename,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f'{matrix_type}_matrix.csv'
    )



#region Logic for adding entries even though they still dont persist... need to use dcc.store i think...

@app.callback(
    Output('add-ped-missing-parents-table', 'children'),
    Input('add-ped-missing-rows-store', 'data')
)
def show_missing_parents_table(invalid_rows):
    """
    Displays an editable table of invalid rows (with missing or unknown parents),
    and shows the "original" parent name if it wasn't recognized in the dataset.
    """
    # Always create a Save Corrections button so its ID is in the layout
    save_button = html.Button(
        "Save Corrections",
        id="add-ped-save-corrections-btn",
        style=CUSTOM_CSS['button']
    )

    if not invalid_rows:
        return html.Div([
            "No invalid rows. All good!",
            html.Br(),
            html.Div(save_button, style={'display': 'none'})  # Hide or disable if you want
        ])

    existing_lines = df['LineName'].unique().tolist()

    rows_html = []
    for i, row_dict in enumerate(invalid_rows):
        row_ln = row_dict.get('LineName','')
        row_mp = row_dict.get('MaleParent','')
        row_fp = row_dict.get('FemaleParent','')

        male_in_dataset = row_mp in existing_lines if row_mp else True
        female_in_dataset = row_fp in existing_lines if row_fp else True

        # Our normal dropdown for male parent
        male_dd = dcc.Dropdown(
            id={'type': 'missing-parent-male', 'index': i},
            options=[{'label': x, 'value': x} for x in existing_lines],
            value=row_mp if male_in_dataset else None,
            placeholder='Select/Correct Male Parent',
            style={'width': '200px'},
        )

        # We'll build the cell contents for the male parent
        # so we can append a "Invalid: original_name" if it’s not recognized
        male_cell_contents = [male_dd]
        if not male_in_dataset and row_mp:
            male_cell_contents.append(
                html.Span(
                    f" (Invalid: {row_mp})",
                    style={'color': 'red', 'marginLeft': '8px'}
                )
            )

        # Our normal dropdown for female parent
        female_dd = dcc.Dropdown(
            id={'type': 'missing-parent-female', 'index': i},
            options=[{'label': x, 'value': x} for x in existing_lines],
            value=row_fp if female_in_dataset else None,
            placeholder='Select/Correct Female Parent',
            style={'width': '200px'},
        )

        # Similarly, note the invalid female name if needed
        female_cell_contents = [female_dd]
        if not female_in_dataset and row_fp:
            female_cell_contents.append(
                html.Span(
                    f" (Invalid: {row_fp})",
                    style={'color': 'red', 'marginLeft': '8px'}
                )
            )

        row_div = html.Tr([
            html.Td(row_ln),
            html.Td(male_cell_contents),
            html.Td(female_cell_contents)
        ])
        rows_html.append(row_div)

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("LineName"),
            html.Th("MaleParent"),
            html.Th("FemaleParent")
        ])),
        html.Tbody(rows_html)
    ])

    return html.Div([
        table,
        html.Br(),
        save_button
    ])




def combine_pedigree_with_new_rows(main_df, new_rows):
    # Convert new_rows (list of dicts) to a DataFrame
    if not new_rows:
        return main_df
    new_rows_df = pd.DataFrame(new_rows)
    # Concatenate
    combined_df = pd.concat([main_df, new_rows_df], ignore_index=True)
    return combined_df




@app.callback(
    [
        Output('add-ped-missing-rows-store', 'data'),
        Output('add-ped-rows-store', 'data'),
        Output('add-ped-upload-status', 'children'),
        Output('data-store', 'data', allow_duplicate=True)  # New output to update the main dataset store
    ],
    [
        Input('add-ped-upload', 'contents'),
        Input('add-ped-save-corrections-btn', 'n_clicks'),
    ],
    [
        State('add-ped-upload', 'filename'),
        State({'type': 'missing-parent-male', 'index': ALL}, 'value'),
        State({'type': 'missing-parent-female', 'index': ALL}, 'value'),
        State('add-ped-missing-rows-store', 'data'),
        State('add-ped-rows-store', 'data'),
    ],
    prevent_initial_call=True
)
def handle_upload_and_save_corrections(upload_contents, save_clicks,
                                       filename,
                                       male_values, female_values,
                                       invalid_rows, valid_rows):
    global df  # we continue to use df as our working global dataset

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Branch for file upload
    if triggered_id == 'add-ped-upload':
        if not upload_contents:
            raise PreventUpdate

        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)

        if not filename.lower().endswith('.txt'):
            return dash.no_update, dash.no_update, "Please upload a .txt file (tab-delimited).", dash.no_update

        try:
            new_df = pd.read_csv(BytesIO(decoded), sep='\t', encoding='latin-1')
        except Exception as e:
            return dash.no_update, dash.no_update, f"Error reading file: {e}", dash.no_update

        required_cols = {'LineName', 'FemaleParent', 'MaleParent'}
        if not required_cols.issubset(new_df.columns):
            msg = "File must have columns: LineName, FemaleParent, MaleParent"
            return dash.no_update, dash.no_update, msg, dash.no_update

        # Use current df to validate; you may choose to use the store instead
        existing_lines = set(df['LineName'].unique())
        found_invalid_rows = []
        found_valid_rows = []
        for _, row in new_df.iterrows():
            ln = str(row['LineName']).strip()
            mp = str(row['MaleParent']).strip() if pd.notna(row['MaleParent']) else ''
            fp = str(row['FemaleParent']).strip() if pd.notna(row['FemaleParent']) else ''
            if ln == '':
                continue
            male_ok = (mp == '') or (mp in existing_lines)
            female_ok = (fp == '') or (fp in existing_lines)
            if male_ok and female_ok:
                found_valid_rows.append(row.to_dict())
            else:
                found_invalid_rows.append(row.to_dict())

        msg = f"Uploaded {filename}. Valid: {len(found_valid_rows)}; Invalid: {len(found_invalid_rows)}."
        # In this branch we are not yet updating the main dataset.
        return (found_invalid_rows, found_valid_rows, msg, dash.no_update)

    # Branch for saving corrections
    elif triggered_id == 'add-ped-save-corrections-btn':
        if not invalid_rows:
            return dash.no_update, dash.no_update, "No invalid rows to correct.", dash.no_update

        # Apply corrections from the dropdown inputs
        for i, row_dict in enumerate(invalid_rows):
            corrected_male = male_values[i] or ''
            corrected_female = female_values[i] or ''
            row_dict['MaleParent'] = corrected_male
            row_dict['FemaleParent'] = corrected_female

        updated_valid_rows = valid_rows + invalid_rows

        # Create a DataFrame from the new entries and update the global dataset
        new_entries_df = pd.DataFrame(updated_valid_rows)
        df = pd.concat([df, new_entries_df], ignore_index=True)

        # Update the dcc.Store by converting the updated DataFrame to a dict of records
        updated_store = df.to_dict('records')
        return ([], updated_valid_rows, "Corrections saved successfully!", updated_store)

    else:
        raise PreventUpdate


#endregion



@app.callback(
    Output('download-subset-link', 'href'),
    [Input('subset-dropdown', 'value')]
)
def update_subset_matrix_download_link(subset_values):
    if not subset_values or current_amatrix is None:
        return dash.no_update
    subset_amatrix = current_amatrix.loc[subset_values, subset_values]
    file_name = f"subset_matrix_{int(time.time())}.csv"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    subset_amatrix.to_csv(file_path)
    safe_filename_subset = urllib.parse.quote(file_path)
    return f'/download?filename={safe_filename_subset}&type=subset'


@app.callback(
    Output('download-subset-link-dummy', 'href'),
    [Input('subset-dropdown-dummy', 'value')]
)
def update_subset_matrix_download_link_dummy(subset_values):
    if not subset_values or current_amatrix is None:
        return dash.no_update
    subset_amatrix = current_amatrix.loc[subset_values, subset_values]
    file_name = f"dummy_subset_matrix_{int(time.time())}.csv"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    subset_amatrix.to_csv(file_path)
    safe_filename_subset = urllib.parse.quote(file_path)
    return f'/download?filename={safe_filename_subset}&type=subset'


@app.callback(
    Output('selected-progeny-modules', 'children'),
    [Input('progeny-module-dropdown', 'value')]
)
def display_selected_progeny_modules(selected_functions):
    """Dynamically display the relevant input sections for the selected function(s)."""
    if not selected_functions:
        return []

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
            html.Button(
                "Find Progeny",
                id="find-progeny-button",
                className="btn btn-success",
                style=CUSTOM_CSS['button']
            ),
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
            html.Button(
                "Find Progeny",
                id="find-single-parent-progeny-button",
                className="btn btn-info",
                style=CUSTOM_CSS['button']
            ),
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
            html.Button(
                "Generate Family Tree",
                id="generate-family-tree-button",
                className="btn btn-warning",
                style=CUSTOM_CSS['button']
            ),
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
            html.Button(
                "Generate Descendant Tree",
                id="generate-descendant-tree-button",
                className="btn btn-warning",
                style=CUSTOM_CSS['button']
            ),
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
            html.Button(
                "Generate Combined Family Tree",
                id="generate-combined-family-tree-button",
                className="btn btn-primary",
                style=CUSTOM_CSS['button']
            ),
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
            html.Button(
                "Generate Family Tree with Temporary Progeny",
                id="generate-temp-progeny-tree-button",
                className="btn btn-warning",
                style=CUSTOM_CSS['button']
            ),
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

    generations = defaultdict(list)
    max_g1 = max(generations1.keys()) if generations1 else 0
    max_g2 = max(generations2.keys()) if generations2 else 0
    for g in range(max(max_g1, max_g2) + 1):
        generations[g].extend(generations1.get(g, []) + generations2.get(g, []))

    line_colors = {}
    for a in ancestors1:
        line_colors[a] = '#ADD8E6'
    for a in ancestors2:
        if a in line_colors:
            line_colors[a] = '#FFFFE0'  # yellow if in both
        else:
            line_colors[a] = '#FFB6C1'

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
            parent_color = line_colors.get(parent, 'lightgrey')
            child_color = line_colors.get(child, 'lightgrey')
            # if both nodes are yellow, make the edge black
            if parent_color == '#FFFFE0' and child_color == '#FFFFE0':
                color = 'black'
            elif child in ancestors1 or child == line1:
                color = '#ADD8E6'
            elif child in ancestors2 or child == line2:
                color = '#FFB6C1'
            else:
                color = '#a4a4a4'
            dot.edge(parent, child, color=color)
            added_edges.add((parent, child))

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
    temp_df = pd.concat(
        [temp_df, pd.DataFrame([[temp_progeny_name, female_parent, male_parent]],
         columns=temp_df.columns)]
    )

    all_ancestors, relationships, generations = find_ancestors(temp_progeny_name, temp_df)
    all_lines = all_ancestors.union({temp_progeny_name})
    relatives_df = temp_df[temp_df['LineName'].isin(all_lines)]
    kinship_matrix = compute_amatrix_diploid(relatives_df)

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
            return '#ADD8E6'
        elif value <= quantiles[0.5]:
            return '#FFB6C1'
        else:
            return '#FFFFE0'

    dot = graphviz.Digraph(comment='Family Tree with Temporary Progeny')
    dot.attr('node', shape='ellipse', style='filled', fontsize='20')

    for _, row in relatives_df.iterrows():
        line_name = row['LineName']
        if (line_name in kinship_matrix.index and
                temp_progeny_name in kinship_matrix.index):
            kv = kinship_matrix.loc[temp_progeny_name, line_name]
        else:
            kv = 0.0
        label = f"{line_name}\nKinship: {kv:.2f}"

        if line_name == temp_progeny_name:
            color = 'orange'
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


if __name__ == '__main__':
    app.run_server(debug=False)
