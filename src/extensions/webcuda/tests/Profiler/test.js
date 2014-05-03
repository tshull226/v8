load("Profiler.js");

%BeginTiming();
profiler = new Profiler();
profiler.stop("test");
profiler.start("test");
profiler.start("test");
profiler.stop("test");

profiler.print();
