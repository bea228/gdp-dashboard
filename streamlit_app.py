import streamlit as st
import jax.numpy as jnp
from jax import grad
from jax.scipy.stats import norm as jnorm
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from scipy.interpolate import griddata
import datetime as dt



st.set_page_config(
    page_title="Option Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

st.title("Black-Scholes Pricing Model")
with st.sidebar:
    st.title("📈 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/benandresen/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Benjamin, Andresen`</a>', unsafe_allow_html=True)

st.sidebar.header('Pricing Model Inputs')

S = st.sidebar.number_input('Underlying Price', value =100.00)

K = st.sidebar.number_input('Strike Price', value =100.00)
T = st.sidebar.number_input('Time to Expiration(Years)', value =1.00)
sigma = st.sidebar.number_input('Volatility (σ)', value =0.20)
r = st.sidebar.number_input('Risk-Free Interest Rate(also used for IV Surface)', value =0.05)
##########################################
################functions#################
########################################## 
def black_scholes(S, K, T, r, sigma, q=0, otype='call'):#uses jax library so gradient can be taken for greeks
    d1 = (jnp.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*jnp.sqrt(T))
    d2 = d1 -sigma * jnp.sqrt(T)
    if otype == 'call':
        call = S * jnp.exp(-q*T) * jnorm.cdf(d1,0,1) - K * jnp.exp(-r*T) * jnorm.cdf(d2,0,1)
        return call 
    else:
        put = K * jnp.exp(-r*T) * jnorm.cdf(-d2,0,1) - S * jnp.exp(-q * T) * jnorm.cdf(-d1,0,1)
        return put
################################################

def bs_call_price(S, K, T, r, sigma, q=0):#faster version
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price
def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan

    def objective_function(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price

    try:
        implied_vol = brentq(objective_function, 1e-6, 5)
    except (ValueError, RuntimeError):
        implied_vol = np.nan

    return implied_vol
#######################################################
    

#get partial derivatives
jax_delta = grad(black_scholes, argnums=0) #with repsect to underlying price
jax_gamma = grad(grad(black_scholes, argnums=0), argnums=0)
jax_vega = grad(black_scholes, argnums=4)#respect to vol
jax_rho = grad(black_scholes, argnums=3)
jax_theta = grad(black_scholes, argnums = 2)
#calcs
call_value = black_scholes(S, K, T, r, sigma, q=0, otype='call')
put_value = black_scholes(S, K, T, r, sigma, q=0, otype='put')
#call greeks
delta = jax_delta(S, K, T, r, sigma, q=0, otype="call")
gamma = jax_gamma(S, K, T, r, sigma, q=0, otype="call")
theta = -jax_theta(S, K, T, r, sigma, q=0, otype="call")
vega = jax_vega(S, K, T, r, sigma, q=0, otype="call")
rho = jax_rho(S, K, T, r, sigma, q=0, otype="call")
#put greeks
pdelta = jax_delta(S, K, T, r, sigma, q=0, otype="put")
pgamma = jax_gamma(S, K, T, r, sigma, q=0, otype="put")
ptheta = -jax_theta(S, K, T, r, sigma, q=0, otype="put")
pvega = jax_vega(S, K, T, r, sigma, q=0, otype="put")
prho = jax_rho(S, K, T, r, sigma, q=0, otype="put")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="background-color: #90EE90; padding: 20px; text-align: center; border-radius: 10px;">
            <h2 style="color: black;">Call Value</h2>
            <p style="font-size: 24px; color: black;">${call_value:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="background-color: #FF6F61; padding: 20px; text-align: center; border-radius: 10px;">
            <h2 style="color: black;">Put Value</h2>
            <p style="font-size: 24px; color: black;">${put_value:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.title("Option Greeks Table")



st.markdown(
    f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    table, th, td {{
        border: 1px solid black;
    }}
    th, td {{
        padding: 10px;
        text-align: center;
        font-size: 16px; /* Make text larger */
        color: black; /* Font color */
    }}
    th {{
        background-color: #d3d3d3; /* Light gray background for headers */
    }}
    td {{
        background-color: #f5f5f5; /* Light gray background for cells */
    }}
    body {{
        background-color: #d3d3d3; /* Light gray page background */
    }}
    </style>
    </head>
    <body>

    

    <table>
    <tr>
        <th></th>
        <th>Delta(Δ)</th>
        <th>Gamma(Γ)</th>
        <th>Theta(θ)</th>
        <th>Vega(ν)</th>
        <th>Rho(ρ)</th>
    </tr>
    <tr>
        <td>Call</td>
        <td>{delta:.2f}</td>
        <td>{gamma:.2f}</td>
        <td>{theta:.2f}</td>
        <td>{vega:.2f}</td>
        <td>{rho:.2f}</td>
    </tr>
    <tr>
        <td>Put</td>
        <td>{pdelta:.2f}</td>
        <td>{pgamma:.2f}</td>
        <td>{ptheta:.2f}</td>
        <td>{pvega:.2f}</td>
        <td>{prho:.2f}</td>
    </tr>
    </table>

    </body>
    </html>
    """,
    unsafe_allow_html=True
)




###############################
#########IV Graph #############
###############################
st.title('Implied Volatility Surface using Black-Scholes')
st.sidebar.header('IV Surface Inputs')
ticker_symbol = st.sidebar.text_input('Ticker Symbol', value='SPY')

min_strike_pct = st.sidebar.number_input(
    'Minimum Strike Price (% of Spot Price)',
    min_value=50.0,
    max_value=199.0,
    value=70.0,
    step=1.0,
    format="%.1f"
)

max_strike_pct = st.sidebar.number_input(
    'Maximum Strike Price (% of Spot Price)',
    min_value=51.0,
    max_value=200.0,
    value=130.0,
    step=1.0,
    format="%.1f"
)
ticker = yf.Ticker(ticker_symbol)
exp_dates = ticker.options
spot = ticker.history(period='1d')
spot = spot['Close'].iloc[-1]

##########################
####Dataframe Creation####
##########################

opt  = ticker.options
chain = ticker.option_chain()
chains = pd.DataFrame()
expirations = ticker.options
for expiration in expirations:
        # tuple of two dataframes
        opt = ticker.option_chain(expiration)
        
        calls = opt.calls
        calls['optionType'] = "call"
        
        puts = opt.puts
        puts['optionType'] = "put"
        
        #chain = pd.concat([calls, puts])
        chain = calls

        chain['expiration'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)
        
        chains = pd.concat([chains, chain]) #only do calls to reduce time calc for IV

    
chains["daysToExpiration"] = (chains.expiration - dt.datetime.today()).dt.days + 1
chains.drop(['contractSymbol', 'lastTradeDate', 'lastPrice', 'change', 'percentChange', 'volume','openInterest', 'impliedVolatility', 'inTheMoney', 'contractSize', 'currency', 'optionType', 'expiration' ], axis=1, inplace=True)
chains['mid']= (chains['bid'] + chains['ask'])/2
chains['daysToExpiration'] = chains['daysToExpiration']/365
chains = chains[chains['daysToExpiration'] <= 2]##filters to only 2 years out

new_min= spot*(min_strike_pct/100)
new_max = spot* (max_strike_pct/100)
chains = chains[(chains['strike']>= new_min) & (chains['strike'] <= new_max)]


#####calc IV on df#####################
with st.spinner('Calculating implied volatility...'):
    chains['impliedVolatility'] = chains.apply(
                    lambda row: implied_volatility(
                        price=row['mid'],
                        S=spot,
                        K=row['strike'],
                        T=row['daysToExpiration'],
                        r=r,
                    ), axis=1
                )
chains['impliedVolatility'] = chains['impliedVolatility']
chains['impliedVolatility'] = chains['impliedVolatility'].astype(float)

chains.dropna(subset=['impliedVolatility'], inplace=True)

chains['impliedVolatility'] *= 100

chains.sort_values('strike', inplace=True)

############################################



# Your existing data preparation code
Y = chains['strike'].values
y_label = 'Strike Price ($)'
        
           

X = chains['daysToExpiration'].values
Z = chains['impliedVolatility'].values



ti = np.linspace(X.min(), X.max(), 50)
ki = np.linspace(Y.min(), Y.max(), 50)
T, K = np.meshgrid(ti, ki)

Zi = griddata((X, Y), Z, (T, K), method='linear')

Zi = np.ma.array(Zi, mask=np.isnan(Zi))

fig = go.Figure(data=[go.Surface(
    x=T, y=K, z=Zi,
    colorscale='Viridis',
    colorbar_title='Implied Volatility (%)'
)])

fig.update_layout(
    scene=dict(
        xaxis_title='Time to Expiration (years)',
        yaxis_title=y_label,
        zaxis_title='Implied Volatility (%)'
    ),
    autosize=False,
    width=900,
    height=800,
    margin=dict(l=65, r=50, b=65, t=90)
)
st.plotly_chart(fig)










