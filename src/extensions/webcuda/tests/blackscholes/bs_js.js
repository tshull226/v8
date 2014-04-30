///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
function CND(d)
{
    var A1 = 0.31938153;
    var A2 = -0.356563782;
    var A3 = 1.781477937;
    var A4 = -1.821255978;
    var A5 = 1.330274429;
    var RSQRT2PI = 0.39894228040143267793994605993438;

    var K = 1.0 / (1.0 + 0.2316419 * Math.abs(d));

    var cnd = RSQRT2PI * Math.exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
function BlackScholesBodyCPU(
    S, //Stock price
    X, //Option strike
    T, //Option years
    R, //Riskless rate
    V  //Volatility rate
)
{
    var sqrtT = Math.sqrt(T);
    var d1 = (Math.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    var d2 = d1 - V * sqrtT;
    var CNDD1 = CND(d1);
    var CNDD2 = CND(d2);

    //Calculate Call and Put simultaneously
    var expRT = Math.exp(- R * T);
    var callResult   = S * CNDD1 - X * expRT * CNDD2;
    var putResult    = X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1);

    return [callResult, putResult]
}


////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options
////////////////////////////////////////////////////////////////////////////////
function bs_js(
    StockPrice,
    OptionStrike,
    OptionYears,
    Riskfree,
    Volatility,
    optN
)
{
    var CallResult = new Float32Array(optN);
    var PutResult = new Float32Array(optN);

    for (var opt = 0; opt < optN; opt++)
    {
        var result =
        BlackScholesBodyCPU(
            StockPrice[opt],
            OptionStrike[opt],
            OptionYears[opt],
            Riskfree,
            Volatility
        );
        CallResult[opt] = result[0];
        PutResult[opt] = result[1];
    }

    return [CallResult, PutResult];
}
