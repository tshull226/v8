
var imageW = 4096;
var imageH = 4096;
var numIterations = 1;
var KERNEL_RADIUS = 8
var KERNEL_LENGTH = 2 * KERNEL_RADIUS + 1

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

	load("tests/Profiler/Profiler.js")
	var profiler = new Profiler();
	profiler.start("Total");
	runJS(h_Kernel, h_Input, profiler);
	profiler.stop("Total");
	profiler.print();
}

function runJS(h_Kernel, h_Input, profiler){
	print("Starting JavaScript Execution!");
	profiler.start("Allocating host memory");
	var h_Buffer = new Array(imageW * imageH);
	var h_Output = new Array(imageW * imageH);
	profiler.stop("Allocating host memory");
	
	for(var i = 0; i < numIterations; i++){
		convolutionRowJS(h_Buffer, h_Input, h_Kernel, imageW, imageH, KERNEL_RADIUS);
		convolutionColumnJS(h_Output, h_Buffer, h_Kernel, imageW, imageH, KERNEL_RADIUS);
	}

	/*
	for(i = 0; i < imageW * imageH; i++){
		print(h_Output[i]);

	}
	*/

	print("done");
}


////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
function convolutionRowJS(
		h_Dst,
		h_Src,
		h_Kernel,
		imageW,
		imageH,
		kernelR
		)
{
	var y, x, sum, k, d;
	for (y = 0; y < imageH; y++)
		for (x = 0; x < imageW; x++)
		{
			var sum = 0;

			for (k = -kernelR; k <= kernelR; k++)
			{
				d = x + k;

				if (d >= 0 && d < imageW)
					sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k];
			}

			h_Dst[y * imageW + x] = sum;
		}
}



////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
function convolutionColumnJS(
		h_Dst,
		h_Src,
		h_Kernel,
		imageW,
		imageH,
		kernelR
		)
{
	var y, x, sum, k, d;
	for (y = 0; y < imageH; y++)
		for (x = 0; x < imageW; x++)
		{
			sum = 0;

			for (k = -kernelR; k <= kernelR; k++)
			{
				d = y + k;

				if (d >= 0 && d < imageH)
					sum += h_Src[d * imageW + x] * h_Kernel[kernelR - k];
			}

			h_Dst[y * imageW + x] = sum;
		}
}

main();
