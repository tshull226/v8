load("tests/histogram/JavaScript/histogram_data.js");
load("tests/histogram/JavaScript/histogram_js.js");
load("tests/Profiler/Profiler.js");

var DataSize = 64 * 1024;
var SCALE = 8 * 1024;
var BigDataSize = DataSize * SCALE;
var BigData = new Uint8Array(BigDataSize);
for(var i = 0; i < SCALE; i++) {
  for(var j = 0; j < DataSize; j++) {
    BigData[DataSize * i + j] = Data[j];
  }
}

profiler = new Profiler();
profiler.start("Total");
var result_js = histogram_js(BigData, BigDataSize, profiler);
profiler.stop("Total");
profiler.print();

/*var sum = 0;
for(var i = 0; i < 256; i++) {
  print(result[i]);
  sum += result[i];
}*/
