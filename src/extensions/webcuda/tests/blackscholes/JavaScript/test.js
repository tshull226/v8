load("tests/blackscholes/JavaScript/bs_data.js");
load("tests/blackscholes/JavaScript/bs_js.js");
load("tests/Profiler/Profiler.js");

var SCALE = 40960;

profiler = new Profiler();

profiler.start("Allocating host memory");
BigStockPrice = new Float32Array(SCALE * 1024);
BigOptionStrike = new Float32Array(SCALE * 1024);
BigOptionYears = new Float32Array(SCALE * 1024);
profiler.stop("Allocating host memory");

for(var i = 0; i < SCALE; i++) {
  for(var j = 0; j < 1024; j++) {
    BigStockPrice[1024 * i + j] = StockPrice[j];
    BigOptionStrike[1024 * i + j] = OptionStrike[j];
    BigOptionYears[1024 * i + j] = OptionYears[j];
  }
}

//profiler.start("Total");
//var result = bs_js(StockPrice, OptionStrike, OptionYears, 0.02, 0.3, 1024);
var result = bs_js(BigStockPrice, BigOptionStrike, BigOptionYears, 0.02, 0.3, SCALE * 1024, profiler);
//profiler.stop("Total");
profiler.print();

/*for(var i = 0; i < 1024; ++i) {
  print(result[0][i]);
}*/
