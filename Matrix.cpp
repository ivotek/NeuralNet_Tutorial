#include "Matrix.h"
#include <iostream>
#include <cstdlib>

// ---------------------------------------------------------------------
// Costruttore
// ---------------------------------------------------------------------
// Crea una matrice di "r" righe e "c" colonne. "data.resize(...)" alloca
// un vettore di "r" vettori, ciascuno contenente "c" elementi inizializzati
// a 0.0. Dopo questa chiamata la matrice esiste già con la forma giusta,
// pronta per essere riempita (a mano, con randomize(), oppure copiando
// dati da un vettore esterno).
Matrix::Matrix(int r, int c) {
    rows = r;
    cols = c;
    data.resize(rows, std::vector<double>(cols, 0.0));
}

// ---------------------------------------------------------------------
// print()
// ---------------------------------------------------------------------
// Stampa a video il contenuto della matrice, una riga per volta.
// Serve solo per il debug (ad es. per controllare a mano se i pesi o
// gli output hanno senso), non è usata dagli algoritmi di calcolo.
void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// ---------------------------------------------------------------------
// randomize()
// ---------------------------------------------------------------------
// Riempie ogni cella della matrice con un numero casuale compreso
// nell'intervallo [-1, 1].
//
// Spiegazione della formula:
//   rand()            -> intero casuale tra 0 e RAND_MAX
//   rand() / RAND_MAX  -> numero casuale tra 0.0 e 1.0
//   ( ... ) * 2.0      -> numero casuale tra 0.0 e 2.0
//   ( ... ) - 1.0      -> numero casuale tra -1.0 e 1.0
//
// E' fondamentale che pesi e bias partano da valori casuali (e non
// tutti uguali, ad es. tutti zero): se partissero tutti dallo stesso
// valore, i neuroni di uno stesso strato imparerebbero tutti la stessa
// cosa e la rete non riuscirebbe mai a specializzarsi.
void Matrix::randomize() {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
}

// ---------------------------------------------------------------------
// add(const Matrix& n)  ->  somma matrice + matrice
// ---------------------------------------------------------------------
// Somma, elemento per elemento, la matrice "n" alla matrice corrente:
//     this[i][j] = this[i][j] + n[i][j]
// Le due matrici devono avere le stesse dimensioni (non viene
// controllato: è responsabilità di chi chiama la funzione).
// Viene usata, ad esempio, per sommare i bias all'output di un layer,
// oppure per applicare gli "aggiornamenti" (delta) calcolati durante
// l'addestramento ai pesi esistenti.
void Matrix::add(const Matrix& n) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] += n.data[i][j];
        }
    }
}

// ---------------------------------------------------------------------
// add(double n)  ->  somma matrice + scalare
// ---------------------------------------------------------------------
// Somma il numero "n" a TUTTI gli elementi della matrice corrente.
void Matrix::add(double n) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] += n;
        }
    }
}

// ---------------------------------------------------------------------
// multiply(a, b)  ->  prodotto matriciale "righe per colonne" (statico)
// ---------------------------------------------------------------------
// Questa è la moltiplicazione "vera" tra matrici, quella che si studia
// a scuola: ogni elemento del risultato è il prodotto scalare tra una
// riga di "a" e una colonna di "b".
//
// Condizione necessaria: il numero di colonne di "a" deve coincidere
// con il numero di righe di "b" (a.cols == b.rows). Se non coincide,
// la moltiplicazione non ha senso matematico e la funzione stampa un
// errore e restituisce una matrice vuota (0x0).
//
// Il risultato ha dimensioni (a.rows x b.cols).
//
// Nella rete neurale questa operazione serve per calcolare, per ogni
// neurone di uno strato, la somma pesata dei suoi input:
//     output = Pesi * Input
Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows) {
        std::cerr << "Errore: le colonne di A non corrispondono alle righe di B." << std::endl;
        return Matrix(0, 0);
    }

    Matrix result(a.rows, b.cols);
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            double sum = 0;
            // Prodotto scalare tra la riga "i" di a e la colonna "j" di b
            for (int k = 0; k < a.cols; k++) {
                sum += a.data[i][k] * b.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    return result;
}

// ---------------------------------------------------------------------
// multiply(const Matrix& n)  ->  prodotto elemento per elemento (Hadamard)
// ---------------------------------------------------------------------
// ATTENZIONE: questa NON è la moltiplicazione matriciale (quella sopra).
// Qui ogni elemento della matrice corrente viene moltiplicato per il
// corrispondente elemento della matrice "n", posizione per posizione:
//     this[i][j] = this[i][j] * n[i][j]
// Le due matrici devono avere le stesse dimensioni.
//
// Viene usata durante la backpropagation per combinare la derivata
// della sigmoide con l'errore, elemento per elemento.
void Matrix::multiply(const Matrix& n) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] *= n.data[i][j];
        }
    }
}

// ---------------------------------------------------------------------
// multiply(double n)  ->  prodotto matrice * scalare
// ---------------------------------------------------------------------
// Moltiplica ogni elemento della matrice corrente per il numero "n".
// Viene usata, ad esempio, per applicare il "learning rate" (tasso di
// apprendimento) al gradiente calcolato durante l'addestramento.
void Matrix::multiply(double n) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] *= n;
        }
    }
}

// ---------------------------------------------------------------------
// map(func)  ->  applica una funzione a ogni elemento
// ---------------------------------------------------------------------
// Prende in input un puntatore a funzione "func" (una funzione che
// riceve un double e restituisce un double) e la applica a TUTTI gli
// elementi della matrice, sostituendo il valore originale con il
// risultato:
//     this[i][j] = func(this[i][j])
//
// E' il meccanismo con cui viene applicata la funzione di attivazione
// (sigmoide) a tutti i neuroni di un layer con una sola chiamata,
// invece di scrivere un ciclo manuale ogni volta (vedi Layer::feedForward).
void Matrix::map(double (*func)(double)) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val = this->data[i][j];
            this->data[i][j] = func(val);
        }
    }
}

// ---------------------------------------------------------------------
// transpose()  ->  matrice trasposta
// ---------------------------------------------------------------------
// Restituisce una NUOVA matrice con righe e colonne scambiate:
//     result[j][i] = this[i][j]
// Una matrice di dimensioni (rows x cols) diventa quindi (cols x rows).
//
// Nella backpropagation la trasposta serve per "far viaggiare" l'errore
// nella direzione opposta rispetto al feedforward: durante il calcolo
// in avanti i pesi vengono usati come (outputNodes x inputNodes), ma per
// retro-propagare l'errore dal layer di output al layer nascosto serve
// la versione trasposta (inputNodes x outputNodes).
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = this->data[i][j];
        }
    }
    return result;
}

// ---------------------------------------------------------------------
// subtract(a, b)  ->  sottrazione elemento per elemento (statica)
// ---------------------------------------------------------------------
// Calcola result = a - b, elemento per elemento. Le due matrici devono
// avere le stesse dimensioni.
//
// Viene usata per calcolare l'errore della rete:
//     errore = target (valore atteso) - output (valore calcolato dalla rete)
Matrix Matrix::subtract(const Matrix& a, const Matrix& b) {
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return result;
}

// ---------------------------------------------------------------------
// dsigmoid(a)  ->  derivata della sigmoide (statica)
// ---------------------------------------------------------------------
// Calcola la derivata della funzione sigmoide, usando la formula
// semplificata:
//     sigmoide'(x) = sigmoide(x) * (1 - sigmoide(x))
//
// IMPORTANTE: questa formula funziona solo se la matrice "a" in
// ingresso contiene GIA' l'output della sigmoide (cioè valori compresi
// tra 0 e 1), e non i valori "grezzi" prima dell'attivazione. Per
// questo, nel codice, viene sempre chiamata passando l'output di un
// layer (che è già stato "sigmoidato" da Layer::feedForward), mai
// l'input grezzo.
//
// La derivata serve per capire "quanto" e "in che direzione" bisogna
// correggere i pesi durante la backpropagation: più il neurone è
// vicino agli estremi (0 o 1), più la derivata è piccola e quindi più
// piccola sarà la correzione (il neurone è già "saturo" e cambia poco).
Matrix Matrix::dsigmoid(const Matrix& a) {
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result.data[i][j] = a.data[i][j] * (1.0 - a.data[i][j]);
        }
    }
    return result;
}
