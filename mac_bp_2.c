// 誤差逆伝搬法のプログラム。（素子のゲインαを１としている。）
// 訓練用データはファイル training.dat に書き込んでおく。
// 1行に1つの訓練データを記入する。入力ノード値をノード順にスペースで
// 区切って記入した後、その入力に対する目標出力を出力素子順にスペースで
// 区切って記入する。

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define NUM_LEARN 50000			// 学習の繰り返し回数をここで指定する。
#define NUM_SAMPLE 3000			// 訓練データのサンプル数。
#define NUM_INPUT 1			// 入力ノード数。
#define NUM_HIDDEN 3			// 中間層（隠れ層）の素子数。
#define NUM_CON 3				//文脈ニューロンの素子数[名嘉]
#define NUM_OUTPUT 1			// 出力素子数。
#define EPSILON 0.05	 		// 学習時の重み修正の程度を決める。
#define THRESHOLD_ERROR 0.01	// 学習誤差がこの値以下になるとプログラムは停止する。
#define BETA 0.8				// 非線形性の強さ

int tx[NUM_SAMPLE][NUM_INPUT], ty[NUM_SAMPLE][NUM_OUTPUT];			// 訓練データを格納する配列。tx = 入力値：ty = 教師信号
double x[NUM_INPUT+NUM_CON+1], h[NUM_HIDDEN+1], c[NUM_CON], y[NUM_OUTPUT];// 閾値表現用に１つ余分に確保。
double w1[NUM_INPUT+1][NUM_HIDDEN], w2[NUM_HIDDEN+1][NUM_OUTPUT]; 	// 閾値表現用に１つ余分に確保。
double h_back[NUM_HIDDEN+1], y_back[NUM_OUTPUT];	 				// 隠れ素子、出力素子における逆伝搬量。


/* Period parameters */  
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */
#define MIXBITS(u,v) ( ((u) & UMASK) | ((v) & LMASK) )
#define TWIST(u,v) ((MIXBITS(u,v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))

static unsigned long state[N]; /* the array for the state vector  */
static int left = 1;
static int initf = 0;
static unsigned long *next;


/* The Prototype Declaration */
void init_genrand(unsigned long);
void init_by_array(unsigned long [], unsigned long, unsigned long);
void next_state(void);
double genrand_real1(void);
double genrand_real2(void);
double genrand_real3(void);

void ReadData(void);
void InitNet(void);
void Feedforward(int);
void Backward(int);
void ModifyWaits(void);
void PrintResults(int, double);
double CalcError(int);
void bit_maker(void);


int main(int argc, char *argv[])
{
	int    ilearn, isample, i, n;
	double error, max_error;
	unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
	unsigned long seed;	
	seed = strtoul(argv[1], NULL, 10);
	for(n=0; n<NUM_CON; n++)
	{
		c[n] = 0.5;
	}


		/* 関数からデータを読みこむ */
		ReadData();

	// 重み初期値の設定。初期値は全て 0 にしないこと。
	init_by_array(init, length, seed); 
	InitNet();
	
	// 学習の繰り返しループ。
	for(ilearn=0; ilearn<NUM_LEARN; ilearn++)
	{
		if((ilearn % 1000) == 0)
		{
			printf("# of learning : %d\n", ilearn);
		}
			
		// 訓練データに関するループ。
		max_error = 0;
		for(isample=0; isample<NUM_SAMPLE; isample++)
		{
			Feedforward(isample);
			error = CalcError(isample);

			if(error > max_error)
			{
				max_error = error;
			}
			
			//printf("# of learning = %d, training data NO. = %d, error = %f\n", ilearn, isample+1, error);
			if((ilearn % 1000) == 0)
			{
				PrintResults(isample, error);
			}
				
			Backward(isample);
			ModifyWaits();
		}

		if(max_error < THRESHOLD_ERROR)
		{
			break;
		}
	}
	
	printf("\n\n# of learning : %d\n", ilearn);
	for(i=0; i<isample; i++)
	{
		Feedforward(i);
		PrintResults(i, CalcError(i));
	}
	
	return(0);
}


void ReadData(void)
{
	int i, isample;

	for(isample=0; isample<NUM_SAMPLE; isample++)
	{
		i = 0;
		tx[isample][i] = 0;
        if((isample % 3) == 0)
        {
         	tx[isample][i] = rand() % 2;
    	}
    	else if((isample % 3) == 1)
    	{
         	tx[isample][i] = rand() % 2;
    	}
    	else
    	{
           	if((tx[isample-1][i]==1)&&(tx[isample-2][i]==1))
           	{
               	tx[isample][i] = 0;
           	}
          	else if((tx[isample-1][i]==0)&&(tx[isample-2][i]==0))
           	{
             	tx[isample][i] = 0;
        	}
        	else
        	{
                tx[isample][i] = 1;
        	}
		}
	}
	for(isample=0; isample<NUM_SAMPLE; isample++)
	{
		i = 0;
		ty[isample][i] = 0;
    	ty[isample][i] = tx[isample+1][i];
    	if(isample == 2999)
        {
         	ty[isample][i] = tx[0][i];
		}
	}
	/* 読み込んだデータの表示 */
		printf("input data: ");
		for(isample=0; isample<NUM_SAMPLE; isample++)
		{
			i = 0;
			printf("%d", tx[isample][i]);
		}
		printf("\n\n ");
		printf("output data: ");
		for(isample=0; isample<NUM_SAMPLE; isample++)
		{
			i = 0;
			printf("%d", ty[isample][i]);
		}
		printf("\n\n");
	printf("Please press a certain key.\n");
	getchar();	
}

void InitNet(void)
{
	int i, j;
	
	for(i=0; i<NUM_INPUT+1; i++)
	{
		for (j=0; j<NUM_HIDDEN; j++)
		{
			w1[i][j] = genrand_real3() - 0.5;
		}
	}
		
	for (i=0; i<NUM_HIDDEN+1; i++)
	{
		for(j=0; j<NUM_OUTPUT; j++)
		{
			w2[i][j] = genrand_real3() - 0.5;
		}
	}
}

void Feedforward(int isample2)
{
	int i, j;
	double net_input;
	
	// 順方向の動作
	// 訓練データに従って、ネットワークへの入力を設定する
	for(i=0; i<NUM_INPUT+NUM_CON; i++)
	{
		if(i<NUM_INPUT)
		{
			x[i] = tx[isample2][i];
		}
		else
		{
			x[i] = c[i-1];
		}
	}

	// 閾値用に x[NUM_INPUT] = 1.0 とする
	x[NUM_INPUT+NUM_CON] = (double)1.0;
	
	// 隠れ素子値の計算
	for(j=0; j<NUM_HIDDEN; j++)
	{
		net_input = 0;
		for(i=0; i<NUM_INPUT+NUM_CON; i++)
		{
			net_input = net_input + w1[i][j] * x[i];
		}

		h[j] = (double)(1.0 / (1.0 + exp((double)net_input * -BETA)));

		// 文脈ニューロン素子値[名嘉]
		c[j] = h[j];
	}
	h[NUM_HIDDEN] = (double)1.0;

	// 出力素子値の計算。
	for(j=0; j<NUM_OUTPUT; j++)
	{
		net_input = 0;

		for(i=0; i<NUM_HIDDEN+1; i++)
		{
			net_input = net_input + w2[i][j] * h[i];
		}
		y[j] = (double)(1.0 / (1.0 + exp((double)net_input * -BETA)));
	}
}

void Backward(int isample2)
{
	int i, j;
	double net_input;
	
	// 逆方向の動作。
	// 出力層素子の逆伝搬時の動作。
	for(j=0; j<NUM_OUTPUT; j++)
	{
		y_back[j] = BETA * (y[j] - ty[isample2][j]) * ((double)1.0 - y[j]) * y[j];
	}

	// 隠れ層素子の逆伝搬時の動作。
	for(i=0; i<NUM_HIDDEN; i++)
	{
		net_input = 0;
		for(j=0; j<NUM_OUTPUT; j++)
		{
			net_input = net_input + w2[i][j] * y_back[j];
		}

		h_back[i] = BETA * net_input * ((double)1.0 - h[i]) * h[i];
	}	
}

void ModifyWaits(void)
{
	int i, j;
	double epsilon = (double)EPSILON;
	
	for(i=0; i<NUM_INPUT+1; i++)
	{
		for(j=0; j<NUM_HIDDEN; j++)
		{
			w1[i][j] = w1[i][j] - epsilon * x[i] * h_back[j];
		}
	}

	for(i=0; i<NUM_HIDDEN+1; i++)
	{
		for(j=0; j<NUM_OUTPUT; j++)
		{
			w2[i][j] = w2[i][j] - epsilon * h[i] * y_back[j];
		}
	}	
}

void PrintResults(int isample2, double error2)
{
	int i;
	
	printf("   training data NO. = %d\n", isample2+1);
	printf("      IN: ");
	for(i=0; i<NUM_INPUT; i++)
	{
		printf("%.0lf ", x[i]);
	}
	
	printf("   Trained_OUT: ");
	for(i=0; i<NUM_OUTPUT; i++)
	{
		printf("%d ", ty[isample2][i]);	
	}
	
	printf("   OUT: ");
	for(i=0; i<NUM_OUTPUT; i++)
	{
		printf("%lf ", y[i]);	
	}

	printf("   error = %lf", error2);

	printf("   CON: ");	
	for(i=0; i<NUM_CON; i++)
	{
		printf("%lf ", c[i]);	
	}
	printf("\n");
}

double CalcError(int isample2)
{
	int i;
	double error = 0.0;
	
	for(i=0; i<NUM_OUTPUT; i++)
	{
		error = error + (y[i] - ty[isample2][i]) * (y[i] - ty[isample2][i]);
	}

	error = error / (double)NUM_OUTPUT;
	
	return(error);
}


/* initializes state[N] with a seed */
void init_genrand(unsigned long s)
{
    int j;
    state[0]= s & 0xffffffffUL;
    for (j=1; j<N; j++) {
        state[j] = (1812433253UL * (state[j-1] ^ (state[j-1] >> 30)) + j); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array state[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        state[j] &= 0xffffffffUL;  /* for >32 bit machines */
    }
    left = 1; initf = 1;
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
void init_by_array(init_key, key_length, seed)
unsigned long init_key[], key_length, seed;
{
    int i, j, k;
    init_genrand(seed);
    i=1; j=0;
    k = (N>key_length ? N : key_length);
    for (; k; k--) {
        state[i] = (state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1664525UL))
          + init_key[j] + j; /* non linear */
        state[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i>=N) { state[0] = state[N-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=N-1; k; k--) {
        state[i] = (state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1566083941UL))
          - i; /* non linear */
        state[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i>=N) { state[0] = state[N-1]; i=1; }
    }

    state[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */ 
    left = 1; initf = 1;
}

void next_state(void)
{
    unsigned long *p=state;
    int j;

    /* if init_genrand() has not been called, */
    /* a default initial seed is used         */
    if (initf==0) init_genrand(5489UL);

    left = N;
    next = state;
    
    for (j=N-M+1; --j; p++) 
        *p = p[M] ^ TWIST(p[0], p[1]);

    for (j=M; --j; p++) 
        *p = p[M-N] ^ TWIST(p[0], p[1]);

    *p = p[M-N] ^ TWIST(p[0], state[0]);
}

/* generates a random number on [0,1]-real-interval ==> 0.0<=rand<=1.0 */
double genrand_real1(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return (double)y * (1.0/4294967295.0); 
    /* divided by 2^32-1 */ 
}

/* generates a random number on [0,1)-real-interval ==> 0.0<=rand<1.0 */
double genrand_real2(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return (double)y * (1.0/4294967296.0); 
    /* divided by 2^32 */
}

/* generates a random number on (0,1)-real-interval ==> 0.0<rand<1.0 */
double genrand_real3(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return ((double)y + 0.5) * (1.0/4294967296.0); 
    /* divided by 2^32 */
}

