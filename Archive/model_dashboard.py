"""
Reusable Dash dashboard for classifier performance reporting.

Usage:
    from model_dashboard import run_dashboard

    # results: dict[target_name -> metrics_dict] as produced by the training loop
    # targets: list of target names
    # features_clean: list of feature column names
    run_dashboard(results, targets, features_clean, port=8051)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _build_overview_fig(results):
    overview_df = pd.DataFrame([
        {'Target': t, 'Accuracy': r['accuracy'], 'F1 Macro': r['f1_macro'],
         'F1 Weighted': r['f1_weighted'], 'PR-AUC': r['pr_auc']}
        for t, r in results.items()
    ])
    fig = go.Figure()
    for metric, color in [('Accuracy', '#636EFA'), ('F1 Macro', '#EF553B'),
                           ('F1 Weighted', '#00CC96'), ('PR-AUC', '#AB63FA')]:
        fig.add_trace(go.Bar(
            name=metric, x=overview_df['Target'], y=overview_df[metric],
            text=overview_df[metric].round(3), textposition='outside',
            marker_color=color,
        ))
    fig.update_layout(
        barmode='group', title='Model Performance Overview',
        yaxis=dict(range=[0, 1.1], title='Score'), xaxis_title='Target',
        template='plotly_white', height=400, legend=dict(orientation='h', y=1.12),
    )
    return fig


def _build_cm_figs(results):
    figs = {}
    for t, r in results.items():
        cm = r['confusion_matrix']
        names = r['class_names']
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=names, y=names,
            colorscale='Blues', text=cm, texttemplate='%{text}',
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
        ))
        fig.update_layout(
            title=f'Confusion Matrix — {t}', xaxis_title='Predicted', yaxis_title='Actual',
            template='plotly_white', height=450, yaxis=dict(autorange='reversed'),
        )
        figs[t] = fig
    return figs


def _build_fi_figs(results, features_clean):
    figs = {}
    for t, r in results.items():
        imp = r['feature_importances']
        top_idx = np.argsort(imp)[-20:]
        fi_df = pd.DataFrame({
            'Feature': [features_clean[i] for i in top_idx],
            'Importance': imp[top_idx],
        }).sort_values('Importance')
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Viridis')
        fig.update_layout(
            title=f'Top 20 Feature Importances — {t}',
            template='plotly_white', height=500, showlegend=False,
        )
        figs[t] = fig
    return figs


def _build_pr_figs(results):
    figs = {}
    for t, r in results.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r['hold_pr_recall'], y=r['hold_pr_precision'],
            mode='lines', line=dict(color='#636EFA', width=2),
            name=f'Holdout PR (AUC={r["pr_auc"]:.3f})',
            fill='tozeroy', fillcolor='rgba(99,110,250,0.1)',
        ))
        th = r['threshold']
        th_idx = np.argmin(np.abs(r['hold_pr_thresholds'] - th))
        fig.add_trace(go.Scatter(
            x=[r['hold_pr_recall'][th_idx]], y=[r['hold_pr_precision'][th_idx]],
            mode='markers', marker=dict(size=14, color='red', symbol='x'),
            name=f'Threshold={th}',
        ))
        prec_target = r['recall_at_prec_target']
        recall_at = r['recall_at_prec']
        if recall_at > 0:
            fig.add_trace(go.Scatter(
                x=[recall_at], y=[prec_target],
                mode='markers+text', marker=dict(size=12, color='#2ecc71', symbol='diamond'),
                name=f'Recall@{prec_target}prec={recall_at:.3f}',
                text=[f'  {recall_at:.3f}'], textposition='middle right',
                textfont=dict(size=12, color='#2ecc71'),
            ))
            fig.add_hline(y=prec_target, line_dash='dash', line_color='#2ecc71',
                          opacity=0.5, annotation_text=f'Precision={prec_target}',
                          annotation_position='top left')
        fig.update_layout(
            title=f'Precision-Recall Curve (Holdout, Fail class) — {t}  |  PR-AUC = {r["pr_auc"]:.3f}',
            xaxis_title='Recall', yaxis_title='Precision',
            template='plotly_white', height=420,
            xaxis=dict(range=[0, 1.05]), yaxis=dict(range=[0, 1.05]),
        )
        figs[t] = fig
    return figs


def _build_sweep_figs(results):
    figs = {}
    for t, r in results.items():
        sweep_df = pd.DataFrame(r['sweep_results'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep_df['threshold'], y=sweep_df['cost'],
            mode='lines', line=dict(color='#EF553B', width=2),
            name='Business Cost', yaxis='y1',
        ))
        fig.add_trace(go.Scatter(
            x=sweep_df['threshold'], y=sweep_df['recall'],
            mode='lines', line=dict(color='#00CC96', width=2, dash='dash'),
            name='Recall (Fail)', yaxis='y2',
        ))
        fig.add_trace(go.Scatter(
            x=sweep_df['threshold'], y=sweep_df['precision'],
            mode='lines', line=dict(color='#636EFA', width=2, dash='dot'),
            name='Precision (Fail)', yaxis='y2',
        ))
        sel_th = r['threshold']
        sel_row = next(s for s in r['sweep_results'] if round(s['threshold'], 2) == sel_th)
        fig.add_trace(go.Scatter(
            x=[sel_th], y=[sel_row['cost']],
            mode='markers', marker=dict(size=14, color='red', symbol='star'),
            name=f'Selected (th={sel_th})', yaxis='y1',
        ))
        fig.update_layout(
            title=f'Threshold Sweep — {t} (FN×{r["fn_cost"]} + FP×{r["fp_cost"]})',
            xaxis_title='Decision Threshold',
            yaxis=dict(title='Business Cost', side='left', showgrid=True),
            yaxis2=dict(title='Recall / Precision', side='right', overlaying='y',
                        range=[0, 1.05], showgrid=False),
            template='plotly_white', height=420,
            legend=dict(orientation='h', y=1.15),
        )
        figs[t] = fig
    return figs


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _make_report_table(report):
    rows = []
    for cls, metrics in report.items():
        if cls in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        rows.append({
            'Class': cls,
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-Score': f"{metrics['f1-score']:.3f}",
            'Support': int(metrics['support']),
        })
    for avg in ('macro avg', 'weighted avg'):
        if avg in report:
            m = report[avg]
            rows.append({
                'Class': avg,
                'Precision': f"{m['precision']:.3f}",
                'Recall': f"{m['recall']:.3f}",
                'F1-Score': f"{m['f1-score']:.3f}",
                'Support': int(m['support']),
            })
    return rows


def _make_sweep_table(sweep_results, selected_threshold):
    key_ths = set(range(0, len(sweep_results), 5))
    for i, s in enumerate(sweep_results):
        if round(s['threshold'], 2) == selected_threshold:
            key_ths.add(i)
    rows = []
    for i in sorted(key_ths):
        s = sweep_results[i]
        fnr = s['FN'] / (s['FN'] + s['TP']) if (s['FN'] + s['TP']) > 0 else 0
        rows.append({
            'Threshold': f"{s['threshold']:.2f}",
            'Recall': f"{s['recall']:.3f}",
            'Precision': f"{s['precision']:.3f}",
            'FPR': f"{s['FPR']:.3f}",
            'FNR': f"{fnr:.3f}",
            'Cost': int(s['cost']),
            '_selected': round(s['threshold'], 2) == selected_threshold,
        })
    return rows


def _make_cost_card(label, value, sub_text, color='#e74c3c'):
    return html.Div(style={
        'textAlign': 'center', 'padding': '15px 20px', 'borderRadius': '8px',
        'border': f'2px solid {color}', 'backgroundColor': '#fff',
        'minWidth': '150px',
    }, children=[
        html.Div(label, style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '4px'}),
        html.Div(f"{value}" if isinstance(value, str) else f"{value:,}",
                 style={'fontSize': '26px', 'fontWeight': 'bold', 'color': color}),
        html.Div(sub_text, style={'fontSize': '11px', 'color': '#95a5a6', 'marginTop': '4px'}),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dashboard(results, targets, features_clean, *,
                  title='Model Performance Report',
                  subtitle='Threshold selected by business cost on validation set',
                  port=8051, debug=False, jupyter_mode='inline'):
    """Launch an interactive Dash dashboard for classifier evaluation.

    Parameters
    ----------
    results : dict
        ``{target_name: metrics_dict}`` as produced by the training loop.
        Each metrics_dict must contain the keys used in the original notebook
        (accuracy, f1_macro, confusion_matrix, feature_importances, etc.).
    targets : list[str]
        Ordered list of target names (used for dropdown default).
    features_clean : list[str]
        Feature column names matching the order of ``feature_importances``.
    title / subtitle : str
        Dashboard header text.
    port : int
        HTTP port for the Dash server.
    debug : bool
        Dash debug mode.
    jupyter_mode : str
        How Dash runs inside Jupyter ('inline', 'external', 'tab', …).
    """
    app = Dash(__name__)

    # Pre-compute all figures
    fig_overview = _build_overview_fig(results)
    cm_figs = _build_cm_figs(results)
    fi_figs = _build_fi_figs(results, features_clean)
    pr_figs = _build_pr_figs(results)
    sweep_figs = _build_sweep_figs(results)

    target_options = [{'label': t, 'value': t} for t in targets]

    app.layout = html.Div(style={
        'fontFamily': 'Segoe UI, Arial, sans-serif', 'padding': '20px',
        'backgroundColor': '#f8f9fa',
    }, children=[
        html.H1(title, style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '5px'}),
        html.P(subtitle, style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),

        # Overview
        html.Div(style={'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px',
                         'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'marginBottom': '25px'}, children=[
            dcc.Graph(figure=fig_overview),
        ]),

        # Target selector
        html.Div(style={'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px',
                         'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'marginBottom': '25px'}, children=[
            html.Label('Select Target:', style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '8px'}),
            dcc.Dropdown(id='target-select', options=target_options, value=targets[0],
                         clearable=False, style={'width': '350px'}),
        ]),

        # KPI cards
        html.Div(id='cost-cards', style={'display': 'flex', 'gap': '15px', 'marginBottom': '25px',
                                          'justifyContent': 'center', 'flexWrap': 'wrap'}),

        # Confusion matrix + classification report
        html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '25px'}, children=[
            html.Div(style={'flex': '1', 'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px',
                             'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'}, children=[
                dcc.Graph(id='confusion-matrix'),
            ]),
            html.Div(style={'flex': '1', 'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px',
                             'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'}, children=[
                html.H3('Classification Report', style={'color': '#2c3e50', 'marginTop': '0'}),
                html.Div(id='report-table'),
            ]),
        ]),

        # PR curve
        html.Div(style={'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px',
                         'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'marginBottom': '25px'}, children=[
            dcc.Graph(id='pr-curve'),
        ]),

        # Threshold sweep: cost curve + table
        html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '25px'}, children=[
            html.Div(style={'flex': '1.2', 'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px',
                             'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'}, children=[
                dcc.Graph(id='sweep-cost-curve'),
            ]),
            html.Div(style={'flex': '0.8', 'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px',
                             'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'overflowY': 'auto',
                             'maxHeight': '460px'}, children=[
                html.H3('Threshold Sweep Table', style={'color': '#2c3e50', 'marginTop': '0'}),
                html.Div(id='sweep-table'),
            ]),
        ]),

        # Feature importance
        html.Div(style={'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px',
                         'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'}, children=[
            dcc.Graph(id='feature-importance'),
        ]),
    ])

    @app.callback(
        Output('confusion-matrix', 'figure'),
        Output('feature-importance', 'figure'),
        Output('report-table', 'children'),
        Output('pr-curve', 'figure'),
        Output('sweep-cost-curve', 'figure'),
        Output('sweep-table', 'children'),
        Output('cost-cards', 'children'),
        Input('target-select', 'value'),
    )
    def update_dashboard(selected_target):
        r = results[selected_target]

        # Classification report table
        rows = _make_report_table(r['report'])
        header = html.Tr([html.Th(c, style={'padding': '10px 14px', 'borderBottom': '2px solid #dee2e6',
                                             'backgroundColor': '#f1f3f5', 'color': '#495057'})
                           for c in ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']])
        body = [
            html.Tr([
                html.Td(row[c], style={
                    'padding': '8px 14px', 'borderBottom': '1px solid #eee',
                    'fontWeight': 'bold' if row['Class'] in ('macro avg', 'weighted avg') else 'normal',
                    'backgroundColor': '#f8f9fa' if row['Class'] in ('macro avg', 'weighted avg') else 'white',
                }) for c in ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
            ]) for row in rows
        ]
        table = html.Table([html.Thead(header), html.Tbody(body)],
                           style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '14px'})

        # Sweep table
        sweep_rows = _make_sweep_table(r['sweep_results'], r['threshold'])
        sweep_cols = ['Threshold', 'Recall', 'Precision', 'FPR', 'FNR', 'Cost']
        sweep_header = html.Tr([html.Th(c, style={'padding': '8px 10px', 'borderBottom': '2px solid #dee2e6',
                                                   'backgroundColor': '#f1f3f5', 'color': '#495057',
                                                   'fontSize': '13px'})
                                 for c in sweep_cols])
        sweep_body = [
            html.Tr([
                html.Td(row[c], style={
                    'padding': '6px 10px', 'borderBottom': '1px solid #eee', 'fontSize': '13px',
                    'fontWeight': 'bold' if row['_selected'] else 'normal',
                    'backgroundColor': '#fff3cd' if row['_selected'] else 'white',
                }) for c in sweep_cols
            ]) for row in sweep_rows
        ]
        sweep_tbl = html.Table([html.Thead(sweep_header), html.Tbody(sweep_body)],
                               style={'width': '100%', 'borderCollapse': 'collapse'})

        # KPI cards
        prec_tgt = r['recall_at_prec_target']
        cost_cards = [
            _make_cost_card('Holdout Cost', r['holdout_cost'],
                            f'FN={r["holdout_fn"]}×{r["fn_cost"]} + FP={r["holdout_fp"]}×{r["fp_cost"]}',
                            color='#e74c3c'),
            _make_cost_card('PR-AUC', f"{r['pr_auc']:.3f}",
                            'Holdout, fail class', color='#AB63FA'),
            _make_cost_card(f'Recall@{prec_tgt}prec', f"{r['recall_at_prec']:.3f}",
                            f'Max recall at precision≥{prec_tgt}', color='#2ecc71'),
            _make_cost_card('Missed Fails (FN)', r['holdout_fn'],
                            f'Cost: {r["holdout_fn"] * r["fn_cost"]:,}', color='#c0392b'),
            _make_cost_card('False Alarms (FP)', r['holdout_fp'],
                            f'Cost: {r["holdout_fp"] * r["fp_cost"]:,}', color='#f39c12'),
            _make_cost_card('Threshold', f"{r['threshold']:.2f}",
                            'Selected on val set', color='#2980b9'),
        ]

        return (cm_figs[selected_target], fi_figs[selected_target], table,
                pr_figs[selected_target], sweep_figs[selected_target], sweep_tbl, cost_cards)

    app.run(jupyter_mode=jupyter_mode, debug=debug, port=port)
