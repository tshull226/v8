var imageW = 4096;
var imageH = 4096;
var numIterations = 1;
var KERNEL_RADIUS = 8
var KERNEL_LENGTH = 2 * KERNEL_RADIUS + 1

load("tests/image_convolution/WebCUDA/convolutionJS.js");
load("tests/image_convolution/WebCUDA/convolutionCUDA.js");

function generateRandom(size){
	var output = new Array(size);
	var i;
	for(i = 0; i < size; i++){
		output[i] = (Math.random() % 16);
	}
	return output;
}

function main(){
	var h_Kernel = generateRandom(KERNEL_LENGTH);
	var h_Input = generateRandom(imageW * imageH);

	load("tests/Profiler/Profiler.js");
	profiler = new Profiler();
	profiler.start("Total");
	outputCUDA = runCUDA(h_Kernel, h_Input);
	profiler.stop("Total");
	profiler.print();

	profiler = new Profiler();
	profiler.start("Total");
	outputJS = runJS(h_Kernel, h_Input);
	profiler.stop("Total");
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
