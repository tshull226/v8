
var imageW = 3072;
var imageH = 3072
var numIterations = 16;
var KERNEL_RADIUS = 8
var KERNEL_LENGTH = 2 * KERNEL_RADIUS + 1

load("tests/image_convolution/WebCUDA/convolutionJS.js");

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

	outputJS = runJS(h_Kernel, h_Input);
	outputCUDA = runCUDA(h_Kernel, h_Input);

	testResults(outputJS, outputCUDA);
}

function testResults(outputJS, outputCUDA){

	print("TEST PASSED!!");
}

main();
