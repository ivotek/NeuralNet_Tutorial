#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

// =====================================================================
// Classe Matrix
// =====================================================================
// Rappresenta una matrice bidimensionale di numeri con la virgola
// (double). E' il "mattoncino" matematico su cui e' costruita tutta
// la rete neurale: sia gli input, sia i pesi, sia i bias, sia gli
// output di ogni strato vengono rappresentati come oggetti Matrix.
//
// Una rete neurale, in fondo, non fa altro che moltiplicare matrici,
// sommare matrici (i bias) e applicare una funzione non lineare
// (la sigmoide) elemento per elemento. Questa classe fornisce tutte
// le operazioni necessarie per fare questo.
// =====================================================================
class Matrix
{
public:
    int rows; // Numero di righe della matrice
    int cols; // Numero di colonne della matrice

    // std::vector è un array dinamico. Un vettore di vettori di double
    // crea essenzialmente una tabella (griglia) di numeri con la virgola.
    // data[i][j] è l'elemento alla riga i, colonna j.
    std::vector<std::vector<double>> data;

    // -----------------------------------------------------------------
    // Costruttore: crea una matrice di r righe e c colonne, con tutti
    // gli elementi inizializzati a 0.0.
    // -----------------------------------------------------------------
    Matrix(int r, int c);

    // Stampa la matrice a video, riga per riga, separando i valori con
    // uno spazio. Utile per il debug.
    void print() const;

    // Riempie la matrice con numeri casuali compresi tra -1 e 1.
    // Viene usata per inizializzare i pesi e i bias di ogni strato
    // della rete prima dell'addestramento.
    void randomize();

    // Somma, elemento per elemento, la matrice "n" alla matrice
    // corrente (this += n). Le due matrici devono avere le stesse
    // dimensioni.
    void add(const Matrix& n);

    // Somma lo scalare "n" a tutti gli elementi della matrice corrente.
    void add(double n);

    // Prodotto "righe per colonne" tra due matrici (moltiplicazione
    // matriciale classica, non elemento per elemento).
    // Condizione necessaria: il numero di colonne di "a" deve essere
    // uguale al numero di righe di "b" (a.cols == b.rows).
    // Il risultato ha dimensioni (a.rows x b.cols).
    // E' "static" perché non modifica nessuna matrice esistente: ne
    // crea e restituisce una nuova.
    static Matrix multiply(const Matrix& a, const Matrix& b);

    // Moltiplicazione elemento per elemento (detta anche "Hadamard
    // product") tra la matrice corrente e "n": this[i][j] *= n[i][j].
    // Attenzione: NON è il prodotto matriciale, è diverso dal metodo
    // statico multiply() sopra!
    void multiply(const Matrix& n);

    // Moltiplica tutti gli elementi della matrice corrente per lo
    // scalare "n".
    void multiply(double n);

    // Applica la funzione "func" a ciascun elemento della matrice,
    // sostituendo il valore originale con il risultato della funzione.
    // Viene usata per applicare la funzione di attivazione (sigmoide)
    // a tutti i neuroni di uno strato in un colpo solo.
    void map(double (*func)(double));

    // Restituisce la trasposta della matrice corrente, cioè una nuova
    // matrice con righe e colonne scambiate (result[j][i] = this[i][j]).
    // Serve durante la backpropagation per "invertire" il verso in cui
    // viaggiano i pesi (dal layer di output verso l'input).
    Matrix transpose() const;

    // Sottrazione elemento per elemento tra due matrici: result = a - b.
    // Usata per calcolare l'errore (target - output ottenuto).
    static Matrix subtract(const Matrix& a, const Matrix& b);

    // Calcola la derivata della funzione sigmoide, applicata a una
    // matrice che contiene GIA' valori sigmoidati (cioè output di
    // un layer). La formula usata è quella semplificata:
    //     sigmoide'(x) = sigmoide(x) * (1 - sigmoide(x))
    // che è valida solo se "a" contiene già gli output della sigmoide
    // e non i valori grezzi prima dell'attivazione. Viene usata nel
    // calcolo del gradiente durante la backpropagation.
    static Matrix dsigmoid(const Matrix& a);
};

#endif // MATRIX_H
