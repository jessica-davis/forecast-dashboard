import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import geopandas as gpd


def create_simple_state_map(selected_state, usa_gpd, fill_color='#3498db'):
    """Create a simple map showing the selected state using Plotly's built-in choropleth"""
    
    df_state = usa_gpd[usa_gpd.STATE_NAME==selected_state]
    df_state['value'] = fill_color
    state_row = df_state[df_state["STATE_NAME"] == selected_state].iloc[0]
    minx, miny, maxx, maxy = state_row.geometry.bounds

    # Create choropleth map
    fig = px.choropleth(
        df_state,
        geojson=df_state.geometry,
        locations=df_state.index,
        color='value',
        hover_name='STATE_NAME'
    )
    fig.update_geos(fitbounds="locations", visible=False)
    
    fig.update_layout(
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            fitbounds="locations",  # zoom to the locations provided
            bgcolor='rgba(0,0,0,0)',  # transparent geo background
            showland=False,
            showlakes=False,
            showcoastlines=False,
            showcountries=False,
            lonaxis=dict(range=[minx, maxx]),  # longitude bounds
            lataxis=dict(range=[miny, maxy]),  # latitude bounds
        ),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
        coloraxis_showscale=False,
        dragmode=False,  # static map
        paper_bgcolor='rgba(0,0,0,0)',  # transparent figure background
        plot_bgcolor='rgba(0,0,0,0)',    # transparent plot area
    )
    
    # Hide hover for other states
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><extra></extra>' # put information here
    )
    
    return fig