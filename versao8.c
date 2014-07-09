/*
   Universidade Federal de Santa Catarina
   Trabalho II de Inteligencia Artificial II
   Disciplina: Inteligência Artificial II
   Nome:       Joildo Schueroff
   Matricula:  10206242
   Data:       24/06/2014

*/
// Bibliotecas de uso do sistema
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/*Valores defines !*/

#define ENTRADA  6
#define OCULTA   6
#define SAIDA    2
#define EPOCA    100000
#define BIAS     1
#define QTD_DA   26
#define VLR_ERRO 0.003
#define ETA      0.5
#define ALPHA    0.9

/*Funções do Sistema*/

double sigmoid(double sig);
double iniciarPesosSinapticos();
double mostrardadosFinais();
double mostrardadosFinais2();
double testarRede();
double backforward();
double feedforward();
double treinarRede();
void carregarDadosTreinamento();
int menu();
void mostrarPesosSinapticos();

//Variaveis Iniciais
int    i, j, k, p, num_epoca, ranpat[QTD_DA+BIAS], epoca;
//Peso das Sinapses
double DeltaPesoOculta[ENTRADA+BIAS][OCULTA+BIAS], DeltaPesoSaida[OCULTA+BIAS][SAIDA+BIAS];
// Variaveis Auxiliares
double erro;
// Dados de Entrada e Saida da Rede
int dadosTreinamento[QTD_DA+BIAS][ENTRADA+BIAS];

float SaidaEsperada[QTD_DA+BIAS][SAIDA+BIAS] =
{

    {  0.010, 0.0015},// A
    {  0.020, 0.0025},// B
    {  0.030, 0.0035},// C
    {  0.040, 0.0045},// D
    {  0.050, 0.0055},// E
    {  0.060, 0.0065},// F
    {  0.070, 0.0075},// G
    {  0.080, 0.0085},// H
    {  0.090, 0.0095},// I
    {  0.100, 0.0000},// J
    {  0.110},// K
    {  0.120},// L
    {  0.130},// M
    {  0.140},// N
    {  0.150},// 0
    {  0.160},// P
    {  0.170},// Q
    {  0.180},// R
    {  0.190},// S
    {  0.200},// T
    {  0.210},// U
    {  0.220},// V
    {  0.230},// W
    {  0.240},// X
    {  0.250},// Y
    {  0.260},// Z
};
//Peso da Saida
double Saidaep[QTD_DA+BIAS][SAIDA+BIAS];

//Variaveis Globais
double SomaO[QTD_DA+BIAS][OCULTA+BIAS],Ocul[QTD_DA+BIAS][OCULTA+BIAS], w_oculta[ENTRADA+BIAS][OCULTA+BIAS];
double SomaS[QTD_DA+BIAS][SAIDA+BIAS], w_saida[OCULTA+BIAS][SAIDA+BIAS];
double DeltaSaida[SAIDA+BIAS], Soma_O_W[OCULTA+BIAS], DeltaOculta[OCULTA+BIAS];

int main()
{
    int retorno;
    printf("Dados do Treinamento carregado....!\n");
    carregarDadosTreinamento();
    printf("Treinando a Rede..................!\n");
    treinarRede();
    printf("\n\n ****Rede Treinada**** \n\n");
    menu();

    return 0;
}


int menu()
{
    int valor;
    while(1)
    {
        printf("\n");
        printf("1- Mostrar dados da Rede..:\n");
        printf("2- Testar a Rede..........:\n");
        printf("3- Pesos Sinapticos.......:\n");
        printf("4- Sair...................:\n");
        scanf("%i",&valor);
        switch(valor)
        {
        case 1:
            printf("Mostrar dados sobre a Rede........!\n");
            mostrardadosFinais();
            break;
        case 2:
            printf("Testar valores na Rede............!\n");
            testarRede();
            break;
        case 3:
            printf("Mostrar os Pesos Sinapticos");
            mostrarPesosSinapticos();
            break;
        case 4:
            printf("Saindo do Sistema da Rede Neural..!\n");
            exit(0);
            break;

        default:
            printf("Valor Invalido! Tente outro valor !\n");
        }
    }
}

//Busca dados do arquivo
void carregarDadosTreinamento()
{
    FILE *p;
    char c;
    int M[QTD_DA+BIAS][ENTRADA+BIAS], i = 0, j = 0;

    p = fopen("arquivo1.txt","r");
    while (!feof(p))
    {
        fscanf(p,"%c",&c);
        //printf("%c",c);
        if(atoi(&c) == 5)
            break;
        if(c == '\n')
        {
            i++;
            j = 0;
        }
        else
        {
            dadosTreinamento[i][j] = atoi(&c);
            j++;
        }
    }
    fclose(p);
}

//Treina a Rede
double treinarRede()
{
    for( epoca = 0; epoca < EPOCA; epoca++)
    {
        erro = 0.0 ;
        for( num_epoca = 1; num_epoca <=QTD_DA; num_epoca++)
        {
            p = num_epoca;

            feedforward();
            backforward();
        }
        if( epoca%5== 0 ) printf("\nEpoca %-5d :   Erro = %f", epoca, erro) ;
        if( erro < VLR_ERRO) break;
    }


}
//Algoritmo de treinamento da E/O e O/S
double feedforward()
{
    for(j = 0; j <= OCULTA; j++)
    {
        SomaO[p][j] = w_oculta[0][j];
        for(i = 0; i <= ENTRADA; i++)
        {
            SomaO[p][j] += dadosTreinamento[p][i] * w_oculta[i][j];
        }
        Ocul[p][j] = sigmoid(SomaO[p][j]);// Função de Ativação da Sigmoid
    }

    for(k = 0; k <= SAIDA; k++)
    {
        SomaS[p][k] = w_saida[0][k];
        for(j = 0; j <= OCULTA; j++)
        {
            SomaS[p][k] += Ocul[p][j] * w_saida[j][k];
        }
        Saidaep[p][k] = sigmoid(SomaS[p][k]); // Função de Ativação da Sigmoid

        // Calculo do Erro para o momentum da rede quando a mesma esta treinada
        erro += 0.5 * (SaidaEsperada[p][k] - Saidaep[p][k]) * (SaidaEsperada[p][k] - Saidaep[p][k]);
        // Propagação do erro Delta para a proxima camada.
        DeltaSaida[k] = (SaidaEsperada[p][k] - Saidaep[p][k]) * Saidaep[p][k] * (1.0 - Saidaep[p][k]);

    }

}
//Algoritmo de treinamento da S/O e O/E
double backforward()
{
    for(j = 0; j <= OCULTA; j++)
    {
        Soma_O_W[j] = 0.0 ;
        for(k = 0; k <= SAIDA; k++)
        {
            Soma_O_W[j] += w_saida[j][k] * DeltaSaida[k];
        }
        DeltaOculta[j] = Soma_O_W[j] * Ocul[p][j] * (1.0 - Ocul[p][j]);
        //Atualização do Delta
    }
    for(j = 0; j <= OCULTA; j++)
    {
        DeltaPesoOculta[0][j] = ETA * DeltaOculta[j] + ALPHA * DeltaPesoOculta[0][j];
        w_oculta[0][j] += DeltaPesoOculta[0][j];
        for( i = 0 ; i <= ENTRADA ; i++)
        {
            DeltaPesoOculta[i][j] = ETA * dadosTreinamento[p][i] * DeltaOculta[j] + ALPHA * DeltaPesoOculta[i][j];
            w_oculta[i][j] += DeltaPesoOculta[i][j] ;
        }
    }
    for( k = 0; k <= SAIDA; k ++)
    {
        DeltaPesoSaida[0][k] = ETA * DeltaSaida[k] + ALPHA * DeltaPesoSaida[0][k] ;
        w_saida[0][k] += DeltaPesoSaida[0][k] ;
        for( j = 0 ; j <= OCULTA ; j++ )
        {
            DeltaPesoSaida[j][k] = ETA * Ocul[p][j] * DeltaSaida[k] + ALPHA * DeltaPesoSaida[j][k] ;
            w_saida[j][k] += DeltaPesoSaida[j][k];
        }
    }
}
//Teste de valores e o funcionamento da rede, juntamente com a tabela ascii
double testarRede()
{
    double e[2];
    int s = 0;
    printf("\n");

    while(s != -1)
    {
        for(i = 0; i < ENTRADA; i++)
        {
            printf("Entrada %i ...............:",i+1);
            scanf("%lf",&e[i]);
            if((e[i] != 0) && (e[i] != 1))
            {
                printf("Erro no valor digitado ! Tente o valor novamente (0)ou(1)");
                return testarRede();
            }
        }
        for(j = 0; j <= ENTRADA; j++)
        {
            for(i = 0; i <=QTD_DA; i++)
            {
                if(dadosTreinamento[i][0] == e[0])
                    if(dadosTreinamento[i][1] == e[1])
                        if(dadosTreinamento[i][2] == e[2])
                            if(dadosTreinamento[i][3] == e[3])
                                if(dadosTreinamento[i][4] == e[4])
                                    if(dadosTreinamento[i][5] == e[5])
                                        break;
            }

        }
        if(i >25)
        {
            printf("Erro no dados digitados na Entrada\n");
            return testarRede();
        }

        printf("Saida Neuronio 1:%c\n",(char)i+65);

        if(i <= 9)
        {
            if(i == 9)
            {
                printf("Saida Neuronio 2:%c\n",((char)i+49-10));
            }
            else
            {
                printf("Saida Neuronio 2:%c\n",(char)i+49);
            }
        }

        printf("Digite -1 para sair do Teste e outro valor para continuar \n");
        scanf("%i",&s);
        if(s == -1)
        {
            return menu();
        }

    }

}
//Mostrar dados gerais da rede.
double mostrardadosFinais()
{
    int cont = 1;
    printf("\n\nREDE TREINADA NA ÉPOCA %d\n",epoca);
    printf("\nINFORMAÇÕES DA REDE NEURAL\n");
    printf("\nCONT\t");
    for( i = 0 ; i < ENTRADA ; i++ )
    {
        printf( "ENT%-2d\t", i+1) ;
    }
    for( k = 0 ; k < SAIDA ; k++ )
    {
        printf( "SAIDA%d\t", k+1) ;
    }
    for( p = 0 ; p < QTD_DA ; p++ )
    {
        printf( "\n%d\t", p);
        for( i = 0 ; i < ENTRADA ; i++ )
        {
            printf( "%-7i ", dadosTreinamento[p][i],p,i) ;
        }
        for(k = 0; k < SAIDA; k++)
        {
            printf( "%c \t ",(char)p+65);
            if(p <= 9)
            {
                if(p == 9)
                {
                    printf( " %c \t",((char)p+49-10));
                }
                else
                {
                    printf( " %c \t",(char)p+49);
                }
            }
            break;
        }
    }
}
//Carrega em 0.0 os pesos sinapticos iniciais
double iniciarPesosSinapticos()
{
    for( j = 0; j < OCULTA ; j++ )
        for( i = 0 ; i < ENTRADA; i++ )
            DeltaPesoOculta[i][j] = 0.0 ;

    for( k = 0 ; k < SAIDA ; k ++ )
        for( j = 0 ; j < OCULTA ; j++ )
            DeltaPesoSaida[j][k] = 0.0;
}
//Mostra os pesos sinapticos da rede depois de treinada
void mostrarPesosSinapticos()
{
    printf("\n");
    for( j = 0 ; j < OCULTA ; j++ )
        for( i = 1 ; i < ENTRADA; i++ )
            printf("Peso E/O[%i][%i] :%lf\n",i, j,Ocul[i][j]);

    printf("\n\n");

    for( k = 0 ; k < SAIDA ; k ++ )
        for( j = 1 ; j < OCULTA ; j++ )
            printf("Peso O/S[%i][%i] :%lf\n",j, k,Saidaep[j][k]);

}

//Função sigmoid que propaga os dados para o próximo estagio da rede.
double sigmoid(double sig)
{
    return  1.0/(1.0 + exp(-sig));
}


