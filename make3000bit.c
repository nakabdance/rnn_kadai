#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int input_bit[3000];
    int output_bit[3000];
    int i, j;
    //シーケンスXORを行うための3000bitの入力信号
    printf("input_bit:");
    for(i=0; i<3000; i++){
        input_bit[i] = 0;
        if((i%3) == 0)
        {
            input_bit[i] = rand() % 2;
        }
        else if((i%3) == 1)
        {
            input_bit[i] = rand() % 2;
        }
        else
        {
            if((input_bit[i-1]==1)&&(input_bit[i-2]==1))
            {
                input_bit[i] = 0;
            }
            else if((input_bit[i-1]==0)&&(input_bit[i-2]==0))
            {
                input_bit[i] = 0;
            }
            else
            {
                input_bit[i] = 1;
            }
        }
        printf("%d", input_bit[i]);

    }
    //教師信号に扱う1bitずつずらしたbit列
    for(j=0; j<3000; j++){
        output_bit[j] = 0;
        output_bit[j] = input_bit[j+1];
        if(j == 2999)
        {
            output_bit[j] = input_bit[0];
        }
    }
}