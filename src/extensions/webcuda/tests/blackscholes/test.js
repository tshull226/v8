load("tests/blackscholes/bs_data.js");
load("tests/blackscholes/bs_webcuda.js");

var result = bs_webcuda(StockPrice, OptionStrike, OptionYears, 0.02, 0.3, 1024);

for(var i = 0; i < 1024; ++i) {
  print(result[0][i]);
}
