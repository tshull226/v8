function histogram_js(Data, byteCount, profiler) {
    profiler.start("Allocating host memory");
    var histogram = new Uint32Array(256);
    profiler.stop("Allocating host memory");

    for (var i = 0; i < 256; i++) {
        histogram[i] = 0;
    }

    profiler.start("Computation");
    for (var i = 0; i < byteCount; i++)
    {
        histogram[Data[i]]++;
    }
    profiler.stop("Computation");

    return histogram;
}
