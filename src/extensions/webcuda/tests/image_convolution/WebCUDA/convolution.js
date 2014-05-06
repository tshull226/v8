var imageW = 8192;
var imageH = 8192;
var numIterations = 1;
var KERNEL_RADIUS = 8
var KERNEL_LENGTH = 2 * KERNEL_RADIUS + 1

load("tests/image_convolution/WebCUDA/convolutionJS.js");
load("tests/image_convolution/WebCUDA/convolutionCUDA.js");

function generateRandom(size, profiler) {
	profiler.start("Allocating host memory");
	var output = new Float32Array(size);
	profiler.stop("Allocating host memory");
	var i;
	for(i = 0; i < size; i++){
		output[i] = (Math.random() % 16);
	}
	return output;
}

function main(){
	load("tests/Profiler/Profiler.js");
	profiler = new Profiler();
	var h_Kernel = generateRandom(KERNEL_LENGTH, profiler);
	var h_Input = generateRandom(imageW * imageH, profiler);
	//profiler.start("Total");
	outputCUDA = runCUDA(h_Kernel, h_Input, profiler);
	//profiler.stop("Total");
	profiler.print();

	profiler = new Profiler();
	//profiler.start("Total");
	outputJS = runJS(h_Kernel, h_Input);
	//profiler.stop("Total");
	profiler.print();

	testResults(outputJS, outputCUDA);
}

function testResults(outputJS, outputCUDA){
	for(var i = 0; i < imageW * imageH; i++){
		if(Math.abs(outputJS[i] - outputCUDA[i]) > 1e-5){
			print("FAILED");
			print("at iteration " + i);
			print("JS Value: " + outputJS[i] + " CUDA Value: " + outputCUDA[i]);
			quit();
		}
	}

	print("TEST PASSED!!");
}

main();
