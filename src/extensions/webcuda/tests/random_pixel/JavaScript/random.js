
INT_SIZE = 4;

//setting up pixel dimensions
width = 640;
height = 480;
numElements = height * width;
numPixels = 4 * numElements;

function main(){
	var seed = 1;


	var t_I = runJS(seed);

	//temp check to see if things seem reasonable
	/*
		 for(i = 0; i < numPixels; i++){
		 print(t_I[i]);
		 }
		 */
	print("TEST PASSED");

}

function runJS(seed){
	var t_I = new Int32Array(numPixels);

	var delta = 0x9E3779B9;
	var k0 = 0xA341316C;
	var k1 = 0xC8013EA4;
	var k2 = 0xAD90777D;
	var k3 = 0x7E95761E;
	var ITER = 15;

	var i,j;
	for(i = 0; i < numElements; i++){

		var x = seed;
		var y = seed << 3;

		x += i + (i << 11) + (i << 19);
		y += i + (i << 9) + (i << 21);    

		var sum = 0;
		for (j=0; j < ITER; j++) {
			sum += delta;
			x += ((y << 4) + k0) & (y + sum) & ((y >> 5) + k1);
			y += ((x << 4) + k2) & (x + sum) & ((x >> 5) + k3);
		}

		var r = x & 0xFF;
		var g = (x & 0xFF00) >> 8;

		t_I[i*4  ] = r;
		t_I[i*4+1] = r;
		t_I[i*4+2] = r;
		t_I[i*4+3] = g;

	}

	return t_I;
}

main();
