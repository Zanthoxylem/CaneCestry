import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import json
import fastcluster
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import flask
import os
import tempfile
from flask import send_file
import io
import pandas as pd
from flask import send_file
from datetime import datetime
import uuid
import dash_table
from dash.exceptions import PreventUpdate
from flask import Flask
from dash import Dash
import numpy as np
import graphviz

server = Flask(__name__)

# Load data
df = pd.read_csv('/home/Zanthoxylum2117/Canecestry/PedInf.txt', sep="\t")

# Global variable to store the current A-matrix
current_amatrix = None

parents_set = set(df['MaleParent'].tolist() + df['FemaleParent'].tolist())

# Identifying rows where LineName is not in the parents set and has no parents
mask = ~df['LineName'].isin(parents_set) & (df['MaleParent'].isna() | df['MaleParent'].isnull()) & (df['FemaleParent'].isna() | df['FemaleParent'].isnull())

# Filtering out the rows based on the mask
filtered_df = df[~mask]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE], server=server)


@app.callback(
    Output('family-tree-image', 'children'),
    [Input('generate-family-tree-button', 'n_clicks')],
    [State('family-tree-dropdown', 'value')]
)
def generate_family_tree(n_clicks, selected_line_name):
    if n_clicks is None or not selected_line_name:
        return dash.no_update

    def find_ancestors(line_name, df, ancestors=None, relationships=None):
        if ancestors is None:
            ancestors = set()
        if relationships is None:
            relationships = []

        current_line = df[df['LineName'] == line_name]
        if not current_line.empty:
            male_parent = current_line.iloc[0]['MaleParent']
            female_parent = current_line.iloc[0]['FemaleParent']
            if pd.notna(male_parent) and male_parent not in ancestors:
                ancestors.add(male_parent)
                relationships.append((male_parent, line_name, 'male'))
                find_ancestors(male_parent, df, ancestors, relationships)
            if pd.notna(female_parent) and female_parent not in ancestors:
                ancestors.add(female_parent)
                relationships.append((female_parent, line_name, 'female'))
                find_ancestors(female_parent, df, ancestors, relationships)
        return ancestors, relationships

    all_ancestors, relationships = find_ancestors(selected_line_name, filtered_df)
    all_lines = all_ancestors.union({selected_line_name})
    relatives_df = filtered_df[filtered_df['LineName'].isin(all_lines)]

    dot = graphviz.Digraph(comment='Family Tree')
    dot.attr('node', shape='ellipse', style='filled')

    for _, row in relatives_df.iterrows():
        line_name = row['LineName']
        label = f"{line_name}"

        # Determine node color based on parental role
        node_color = 'lightgrey'
        if row['LineName'] == selected_line_name:
            node_color = 'green'  # Highlight the selected line

        dot.node(line_name, label=label, fillcolor=node_color, fontcolor='black', color='black')

    for parent, child, role in relationships:
        if role == 'male':
            color = 'blue'
        elif role == 'female':
            color = 'red'
        else:
            color = 'black'
        dot.edge(parent, child, color=color)

    tree_buffer = BytesIO()
    tree_buffer.write(dot.pipe(format='png'))
    tree_buffer.seek(0)
    encoded_tree = base64.b64encode(tree_buffer.read()).decode('utf-8')
    tree_src = f'data:image/png;base64,{encoded_tree}'

    return html.Img(src=tree_src, style=CUSTOM_CSS['image'])

@app.callback(
    Output('descendant-tree-image', 'children'),
    [Input('generate-descendant-tree-button', 'n_clicks')],
    [State('family-tree-dropdown', 'value')]
)
def generate_descendant_tree(n_clicks, selected_line_name):
    if n_clicks is None or not selected_line_name:
        return dash.no_update

    def find_descendants(line_name, df, descendants=None, relationships=None):
        if descendants is None:
            descendants = set()
        if relationships is None:
            relationships = []

        current_line = df[df['MaleParent'] == line_name].append(df[df['FemaleParent'] == line_name])
        if not current_line.empty:
            for _, row in current_line.iterrows():
                child = row['LineName']
                male_parent = row['MaleParent']
                female_parent = row['FemaleParent']
                if child not in descendants:
                    descendants.add(child)
                    relationships.append((line_name, child, 'descendant'))
                    find_descendants(child, df, descendants, relationships)
        return descendants, relationships

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

def find_relatives(line_names, df):
    ancestors, relationships = find_ancestors(line_names, df)
    descendants, _ = find_descendants(line_names, df)
    descendant_ancestors = [find_ancestors([desc], df) for desc in descendants]
    all_relatives = list(set(ancestors + descendants + [item for sublist in descendant_ancestors for item in sublist]))
    return all_relatives

def find_ancestors(line_names, df, processed=None, relationships=None):
    if processed is None:
        processed = set()
    if relationships is None:
        relationships = []

    parents_df = df[df['LineName'].isin(line_names)][['MaleParent', 'FemaleParent']].dropna()
    parents = parents_df['MaleParent'].tolist() + parents_df['FemaleParent'].tolist()

    new_parents = [parent for parent in parents if parent not in processed]

    if not new_parents:
        return line_names, relationships

    processed.update(new_parents)
    all_relatives = list(set(line_names + new_parents))

    for parent in new_parents:
        for line in line_names:
            if parent == df[df['LineName'] == line]['MaleParent'].values[0]:
                relationships.append((parent, line, 'male'))
            if parent == df[df['LineName'] == line]['FemaleParent'].values[0]:
                relationships.append((parent, line, 'female'))

    return find_ancestors(all_relatives, df, processed, relationships)

def find_descendants(line_names, df, processed=None, relationships=None):
    if processed is None:
        processed = set()
    if relationships is None:
        relationships = []

    progeny_df = df[df['MaleParent'].isin(line_names) | df['FemaleParent'].isin(line_names)]
    progeny = progeny_df['LineName'].tolist()

    new_progeny = [child for child in progeny if child not in processed]

    if not new_progeny:
        return line_names, relationships

    processed.update(new_progeny)
    all_relatives = list(set(line_names + new_progeny))

    for child in new_progeny:
        for line in line_names:
            if line == df[df['LineName'] == child]['MaleParent'].values[0]:
                relationships.append((line, child, 'descendant'))
            if line == df[df['LineName'] == child]['FemaleParent'].values[0]:
                relationships.append((line, child, 'descendant'))

    return find_descendants(all_relatives, df, processed, relationships)


CUSTOM_CSS = {
    'container': {'padding': '20px', 'margin-top': '20px', 'background-color': '#f9f9f9'},
    'header': {'textAlign': 'center', 'padding': '10px', 'color': '#2c3e50', 'font-family': 'Arial', 'font-weight': 'bold', 'font-size': '40px'},
    'button': {'margin': '5px', 'font-weight': 'bold', 'background-color': '#2980b9', 'color': 'white'},
    'table': {'overflowX': 'auto', 'margin-bottom': '20px', 'border': '1px solid #ccc'},
    'image': {'width': '100%', 'padding': '10px'},
    'dropdown': {'font-weight': 'bold', 'color': '#2980b9'}
}

def main_page_layout():
    return dbc.Container([
        dbc.Row(dbc.Col(html.H1("CaneCestry 2.0", className="app-header", style=CUSTOM_CSS['header']))),
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
    ], fluid=True, style=CUSTOM_CSS['container'])

def progeny_finder_layout():
    return dbc.Container([
        dbc.Row(dbc.Col(html.H1("Progeny Finder", className="app-header", style=CUSTOM_CSS['header']))),
        dbc.Row(dbc.Col(html.Div(id='progeny-module', children=[
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
        ])), className="mb-4"),

        dbc.Row(dbc.Col(html.Div(id='single-parent-progeny-module', children=[
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
        ])), className="mb-4"),

        dbc.Row(dbc.Col(html.Div(id='family-tree-module', children=[
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
        ]))),
        
        dbc.Row(dbc.Col(html.Div(id='descendant-tree-module', children=[
            html.H4("Generate Descendant Tree"),
            html.Button("Generate Descendant Tree", id="generate-descendant-tree-button", className="btn btn-warning", style=CUSTOM_CSS['button']),
            html.Div(id='descendant-tree-image')
        ]))),

        dbc.Row(dbc.Col(html.A("Back to Main Page", href="/", className="btn btn-secondary", style=CUSTOM_CSS['button']))),
    ], fluid=True, style=CUSTOM_CSS['container'])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/progeny-finder':
        return progeny_finder_layout()
    else:
        return main_page_layout()

@app.callback(
    Output('selected-line-names-list', 'children'),
    [Input('line-name-dropdown', 'value')]
)
def update_selected_line_names_list(selected_line_names):
    if not selected_line_names:
        return []

    list_items = [html.Li(name) for name in selected_line_names]
    return list_items

@app.callback(
    Output('parent-info-table', 'data'),
    [Input('line-name-dropdown', 'value')]
)
def update_parent_info_table(selected_line_names):
    if not selected_line_names:
        return []

    filtered_rows = df[df['LineName'].isin(selected_line_names)][['LineName', 'MaleParent', 'FemaleParent']]
    return filtered_rows.to_dict('records')

@app.callback(
    [Output('heatmap-image', 'src'), Output('download-full-link', 'href'), Output('full-matrix-csv-path', 'children'), Output('matrix-store', 'data'), Output('loading-output', 'children')],
    [Input('generate-amatrix-button', 'n_clicks')],
    [State('line-name-dropdown', 'value')]
)
def generate_amatrix_and_heatmap(n_clicks, selected_line_names):
    global current_amatrix
    if n_clicks is None or not selected_line_names:
        return [dash.no_update] * 5

    loading_indicator = dbc.Spinner(size="lg", color="primary", children=[
        html.Div("Generating A-Matrix and heatmap, please wait...")
    ])

    all_related_lines = find_relatives(selected_line_names, filtered_df)
    relatives_df = filtered_df[filtered_df['LineName'].isin(all_related_lines)]
    sorted_relatives_df = sort_pedigree_df(relatives_df)
    current_amatrix = compute_amatrix_diploid_revised(sorted_relatives_df)

    temp_file_full = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    current_amatrix.to_csv(temp_file_full)
    temp_file_full.close()

    heatmap_plot = sns.clustermap(current_amatrix, method='average', cmap='Spectral', figsize=(15, 15), row_cluster=True, col_cluster=True)
    plt.close(heatmap_plot.fig)

    heatmap_buffer = BytesIO()
    heatmap_plot.savefig(heatmap_buffer, format='png')
    heatmap_buffer.seek(0)
    encoded_heatmap = base64.b64encode(heatmap_buffer.read()).decode('utf-8')
    heatmap_src = f'data:image/png;base64,{encoded_heatmap}'

    filename = "_".join(selected_line_names) + ".csv"
    matrix_id = str(uuid.uuid4())
    in_memory_matrices[matrix_id] = {
        "full_matrix": current_amatrix,
        "filename": filename
    }

    full_matrix_link = f'/download?filename={temp_file_full.name}&type=full'
    store_data = current_amatrix.to_dict()
    full_matrix_csv_path = temp_file_full.name

    return [heatmap_src, full_matrix_link, full_matrix_csv_path, store_data, loading_indicator]

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
    options = [{'label': name, 'value': name} for name in line_names]
    return options

@app.callback(
    Output('subset-matrix-display', 'children'),
    [Input('subset-dropdown', 'value')],
    [State('matrix-store', 'data')]
)
def display_subset_matrix(subset_values, matrix_data):
    if not subset_values or not matrix_data:
        return ""

    current_amatrix = pd.DataFrame(matrix_data)
    subset_amatrix = pd.DataFrame(index=subset_values, columns=subset_values)

    for row in subset_values:
        for col in subset_values:
            if row in current_amatrix.index and col in current_amatrix.columns:
                subset_amatrix.at[row, col] = current_amatrix.at[row, col]
            else:
                subset_amatrix.at[row, col] = 0

    table_header = [html.Thead(html.Tr([html.Th(col) for col in [''] + list(subset_amatrix.columns)]))]
    table_body = [html.Tbody([html.Tr([html.Td(subset_amatrix.index[i])] + [html.Td(subset_amatrix.iloc[i, j]) for j in range(len(subset_amatrix.columns))]) for i in range(len(subset_amatrix))])]
    return html.Table(table_header + table_body, className="table")

@app.server.route('/download')
def download_file():
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

    subset_matrix_link = f'/download?filename={temp_file_subset.name}&type=subset'

    return subset_matrix_link

@app.callback(
    Output('progeny-results', 'children'),
    [Input('find-progeny-button', 'n_clicks')],
    [State('progeny-line-dropdown', 'value'), State('progeny-line-dropdown-2', 'value')]
)
def find_progeny(n_clicks, line1, line2):
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
                f"{row['LineName']} (Female: {Female_parent}, Male: {Male_parent})"
            )
        )

    return html.Ul(results)



def compute_amatrix_diploid_revised(pedigree_df):
    individuals = pedigree_df['LineName'].unique()
    n = len(individuals)
    index_map = {ind: i for i, ind in enumerate(individuals)}
    A = np.identity(n)

    def get_index(ind):
        return index_map.get(ind, -1)

    for i, row in enumerate(pedigree_df.iterrows()):
        _, data = row
        ind_idx = get_index(data['LineName'])
        sire_idx = get_index(data['MaleParent'])
        dam_idx = get_index(data['FemaleParent'])

        if sire_idx == -1 and dam_idx == -1:
            A[ind_idx, ind_idx] = 1
            for j in range(n):
                if j != ind_idx:
                    A[ind_idx, j] = A[j, ind_idx] = 0

        elif sire_idx == -1:
            A[ind_idx, ind_idx] = 1
            for j in range(n):
                if j != ind_idx:
                    A[ind_idx, j] = A[j, ind_idx] = 0.5 * A[dam_idx, j]

        elif dam_idx == -1:
            A[ind_idx, ind_idx] = 1
            for j in range(n):
                if j != ind_idx:
                    A[ind_idx, j] = A[j, ind_idx] = 0.5 * A[sire_idx, j]

        else:
            A[ind_idx, ind_idx] = 1 + 0.5 * A[sire_idx, dam_idx]
            for j in range(n):
                if j != ind_idx:
                    A[ind_idx, j] = A[j, ind_idx] = 0.5 * (A[sire_idx, j] + A[dam_idx, j])

    labeled_A = pd.DataFrame(A, index=individuals, columns=individuals)
    return labeled_A

def sort_pedigree_df(pedigree_df):
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

def calculate_lrf(pedigree_df):
    lrf_values = {line: None for line in pedigree_df['LineName'].unique()}

    def compute_lrf(line):
        if lrf_values[line] is not None:
            return lrf_values[line]

        row = pedigree_df[pedigree_df['LineName'] == line]
        if row.empty or (pd.isna(row['MaleParent'].values[0]) and pd.isna(row['FemaleParent'].values[0])):
            lrf_values[line] = 1
            return 1

        male_parent, female_parent = row['MaleParent'].values[0], row['FemaleParent'].values[0]
        male_lrf = compute_lrf(male_parent) if not pd.isna(male_parent) else 1
        female_lrf = compute_lrf(female_parent) if not pd.isna(female_parent) else 1

        lrf_values[line] = (male_lrf + female_lrf) / 2 / 2
        return lrf_values[line]

    for line in lrf_values.keys():
        compute_lrf(line)

    return lrf_values

if __name__ == '__main__':
    app.run_server(debug=True)