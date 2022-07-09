#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip> // for setw()

// �ܼ�â�� ��İ��� ���
template <typename T>
void print_matrix(std::vector<T> &output, int M, int N, std::string name)
{
    std::cout << "[INFO] Print Matrix : " << name << std::endl;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            std::cout << std::setw(3) << output[m * N + n] << " ";
        }std::cout << std::endl;
    }std::cout << std::endl;
}
// �ܼ�â�� ��İ��� ���
template <typename T>
void print_matrix(std::vector<std::vector<T>> &output, int M, int N, std::string name)
{
    std::cout << "[INFO] Print Matrix : " << name << std::endl;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            std::cout << std::setw(3) << output[m][n] << " ";
        }std::cout << std::endl;
    }std::cout << std::endl;
}
// ������ ���Ϸ� ����
template <typename T>
void tofile(T* Buffer, int data_count, std::string fname = "../valid_py/from_C") 
{
    std::ofstream fs(fname, std::ios::binary);
    if (fs.is_open())
        fs.write((const char*)Buffer, data_count * sizeof(T));
    fs.close();
    std::cout << "[INFO] Done File Production to " << fname << std::endl;
}

// ��� ��� ���� 
void gen_matrix(std::vector<std::vector<float>> &costMatrix, int& N, int& M, bool generate_costMatrix);
// Step 1. �� �࿡�� �ּڰ��� ���� �� ���� ó��
void step1(std::vector<float> &costM, const int H, const int W);
// Step 2. �� ������ �ּڰ��� ���� �� ���� ó��
void step2(std::vector<float> &costM, const int H, const int W);
// Step 3. 0�� ���� ���� �����ϴ� �� �Ǵ� ���� �ּ� ���� Ž��
void step3_0(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W);
void step3(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W);
// step 4. �ּҼ� ��
void step4(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, int H, int W, float maxV, std::vector<std::vector<float>>&assign);
// step 5. ��ó�� �۾�
void step5(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, int H, int W, float maxV, std::vector<std::vector<float>>&assign);
// step 6. ���� ���� �ĺ��� Ž��
void step6(int d, int c, std::vector<std::vector<float>>& map, std::vector<float>& candi, std::vector<int>& check, std::vector<std::vector<float>>& out);
// �밡���� �˰��� ���� (�������� ������ �Լ� ���� ���)
float Solve(const float **Cost, const int N, const int M, const int MODE, float* assignment_index);

int main() 
{
    //�Է� �غ� [N, M] OR [H, W] OR [rows, cols]
    int N = 5;                          // �ο���, ROWS, ��
    int M = 5;                          // �۾���, COLS, ��
    bool generate_costMatrix = false;   // ��� ��� ���� (true : [N, M] ��� ���� ����, false : TEST ���� ��� ���)
    bool valid_py = false;              // ���� ��ũ��Ʈ ���� (true : ���̽� ���� �ڵ� ����)

    std::vector<std::vector<float>> costMatrix;         // ��� ���
    gen_matrix(costMatrix, N, M, generate_costMatrix);  // ��� ��� ����
    print_matrix(costMatrix, N, M, "Cost Matrix");      // ��� ��� ���

    const float **Cost = new const float*[N];
    for (int i = 0; i < N; i++){
        Cost[i] = costMatrix[i].data();
    }

    float *assignment_index_0 = new float[N];
    float *assignment_index_1 = new float[N];

    Solve(Cost, N, M, 0, assignment_index_0);
    Solve(Cost, N, M, 1, assignment_index_1);

    // python ���� ��ũ��Ʈ ����
    if (valid_py) {
        std::cout << "\n==== Python Validation Code ====" << std::endl;
        std::vector<float> costM;
        for (int h = 0; h < N; h++) {
            for (int w = 0; w < M; w++) {
                costM.push_back(Cost[h][w]);
            }
        }
        tofile(costM.data(), N * M, "../valid_py/costMatrix");
        tofile(assignment_index_0, N, "../valid_py/index_0");
        tofile(assignment_index_1, N, "../valid_py/index_1");
        std::string command = "python ../valid_py/valid.py --H=" + std::to_string(N) + " --W=" + std::to_string(M);
        const char *cmd = command.c_str();
        system(cmd);
    }

    // �ڿ�(�޸�) ����
    delete[] Cost;
    delete[] assignment_index_0;
    delete[] assignment_index_1;
    return 0;
}

float Solve(const float **Cost, const int N, const int M, const int MODE, float* assignment_index)
{
    int H = N;
    int W = M;
    // 2D - > 1D
    std::vector<float> costM;
    float maxV = 0;
    float minV = 0;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            float elt = Cost[h][w];
            costM.push_back(elt);
            if (maxV < elt) maxV = elt;
            if (minV > elt) minV = elt;
        }
    }
    //print_matrix(costM, H, W, "cost Matrix 0");

    // ���� ��� ��Ŀ� ���̳ʽ� ���� ������, �ּҰ��� ���밪���� ��� ���� 
    if (minV < 0) {
        for (int i = 0; i < costM.size(); i++) {
            costM[i] = fabs(maxV) + costM[i];
        }
        //print_matrix(costM, H, W, "cost Matrix (remove negative values)");
    }
    
    // ���� �ִ� ��� ��� �̸�, ��� ����� �ִ밪���� ��Ұ��� ����.
    if (MODE == 1) { // MODE == 1 �ִ�ȭ, MODE == 0 �ּ�ȭ
        for (int i = 0; i < costM.size(); i++) {
            costM[i] = maxV - costM[i];
        }
        //print_matrix(costM, H, W, "cost Matrix (change for maximize)");
    }

    //Step 1. �� �࿡�� �ּڰ��� ���� �� ���� ó��
    step1(costM, H, W); // O(N^2)

    //Step 2. �� ������ �ּڰ��� ���� �� ���� ó��
    step2(costM, H, W); // O(N^2)

    // Step 3. ��Ŀ��� 0�� ���� �ּҼ� ���� Ž��
    std::vector<int> coveredCols(W);
    std::vector<int> coveredRows(H);
    step3(costM, coveredCols, coveredRows, H, W); // O(N^4)

    // step 4
    // ã�� �ּҼ� ������ ��� ���ο� ���� �� ���� ���̿� ���ٸ�   -> step 6 ���� ���� �ĺ��� Ž��(������ step)
    // ã�� �ּҼ� ������ ��� ���ο� ���� �� ���� ���̺��� �۴ٸ� -> step 5 ��ó�� �۾� -> step 3 -> step 4
    std::vector<std::vector<float>> candidates;// candidatase
    step4(costM, coveredCols, coveredRows, H, W, maxV, candidates);

    if (candidates.size() != 0) {
        // �ĺ� ���յ��� ��� ���
        std::vector<float> candidates_cost;
        for (int c = 0; c < candidates.size(); c++) {
            float sum = 0;
            for (int h = 0; h < H; h++) {
                sum += Cost[h][int(candidates[c][h])];
            }
            candidates_cost.push_back(sum);
        }

        // MODE�� ���� ���� ��� ����
        float cost_value;
        int cost_index;
        if (MODE == 1) {
            cost_value = *max_element(candidates_cost.begin(), candidates_cost.end());
            cost_index = max_element(candidates_cost.begin(), candidates_cost.end()) - candidates_cost.begin();
        }
        else {
            cost_value = *min_element(candidates_cost.begin(), candidates_cost.end());
            cost_index = min_element(candidates_cost.begin(), candidates_cost.end()) - candidates_cost.begin();
        }

        for (int i = 0; i < H; i++) {
            assignment_index[i] = candidates[cost_index][i];
        }

        // ��� ���
        std::cout << "Mode " << MODE << " �� ��� Total cost : " << cost_value;
        std::cout << ", assignment_index = { ";
        for (int i = 0; i < N; i++) {
            if (i == N - 1) {
                std::cout << assignment_index[i] << " }" << std::endl;
            }
            else {
                std::cout << assignment_index[i] << ", ";
            }
        }
    }
    else {
        std::cout << "[ERROR] Something wrong... " << std::endl; // ������ ���� ó��
    }

    return 0;
}

// ��� ��� ���� 
void gen_matrix(std::vector<std::vector<float>> &costMatrix, int& N, int& M, bool generate_costMatrix) 
{
    if (generate_costMatrix) { // ���� �� ����
        srand(time(0));
        for (int h = 0; h < N; h++) {
            std::vector<float> temp;
            for (int w = 0; w < M; w++) {
                temp.push_back(rand() % 100);
            }
            costMatrix.push_back(temp);
        }
    }
    else {// TEST ���� ���
        N = 3;
        M = 4;
        costMatrix = { {3, 7, 5, 11},
                       {5, 4, 6, 3},
                       {6, 10, 1, 1} };
    }
}

// Step 1. �� �࿡�� �ּڰ��� ���� �� ���� ó��
void step1(std::vector<float> &costM, const int H, const int W) 
{
    //std::cout << "step1" << std::endl;
    for (int h = 0; h < H; h++) {
        // 1-1 �� �ึ�� �ּڰ� ���ϱ�
        float minV = costM[h * W];
        for (int w = 1; w < W; w++) {
            if (minV == 0) break;
            if (minV > costM[h * W + w]) {
                minV = costM[h * W + w];
            }
        }
        if (minV == 0) continue;
        // 1-2 �� ���� �ּڰ����� ���� ��Ұ� ����
        for (int w = 0; w < W; w++) {
            costM[h * W + w] = costM[h * W + w] - minV;
        }
    }
}

// Step 2. �� ������ �ּڰ��� ���� �� ���� ó��
void step2(std::vector<float> &costM, const int H, const int W) 
{
    //std::cout << "step2" << std::endl;
    for (int w = 0; w < W; w++) {
        // 2-1 �� ������ �ּڰ� ���ϱ�
        float minV = costM[w];
        for (int h = 1; h < H; h++) {
            if (minV == 0) break;
            if (minV > costM[h * W + w]) {
                minV = costM[h * W + w];
            }
        }
        if (minV == 0) continue;
        // 2-2 �� ���� �ּڰ����� ���� ��Ұ� ����
        for (int h = 0; h < H; h++) {
            costM[h * W + w] = costM[h * W + w] - minV;
        }
    }
}

// Step 3-0. �ּ� Ž��
void step3_0(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W) 
{
    // ������� ã�� ������ ������ �ʴ� ���� �� ������ ��� �� ���⿡�� ���� Ž�� 
    int zero_count = 0;
    std::vector<int> col_zeros(W);
    std::vector<int> row_zeros(H);
    for (int h = 0; h < H; h++) {
        if (coveredRows[h] == 0) {
            for (int w = 0; w < W; w++) {
                if (coveredCols[w] == 0) {
                    if (costM[h * W + w] == 0) {
                        col_zeros[w]++;
                        row_zeros[h]++;
                        zero_count++;
                    }
                }
            }
        }
    }
    if (zero_count == 0) { // ���� ���� �ִ� ���� ���ٸ�, ����
        // exit
    }
    else {
        int over_one_count = 0; // ���� 2�� �̻� ���Ե� ���� ��
        for (int h = 0; h < H; h++) {
            if (row_zeros[h] > 1) { 
                over_one_count++;
            }
        }
        for (int w = 0; w < W; w++) {
            if (col_zeros[w] > 1) {
                over_one_count++;
            }
        }

        float Col_value = *max_element(col_zeros.begin(), col_zeros.end());
        float Row_value = *max_element(row_zeros.begin(), row_zeros.end());

        if (over_one_count == 0) { // ���� 1���� ���� ���ε鸸 ���� ���
            if (Col_value > Row_value) {
                int Col_index = max_element(col_zeros.begin(), col_zeros.end()) - col_zeros.begin();
                coveredCols[Col_index] = true;
            }
            else {
                int Row_index = max_element(row_zeros.begin(), row_zeros.end()) - row_zeros.begin();
                coveredRows[Row_index] = true;
            }
        }
        else {
            // ���� 2�� �̻� ���� ���ε鰣�� ���� ���� �ּ� �������� ���� ������ Ž���Ͽ� �켱 ����
            std::vector<int> con_btw_col_lines(W);
            for (int w = 0; w < W; w++) {
                int relate_cnt = 0;
                if (col_zeros[w] > 1) {
                    for (int h = 0; h < H; h++) {
                        if (row_zeros[h] > 1) {
                            if (costM[h * W + w] == 0) {
                                relate_cnt++;
                            }
                        }
                    }
                    con_btw_col_lines[w] = relate_cnt;
                }
                else {
                    con_btw_col_lines[w] = INT_MAX; // 0�� �ϳ��� ���� ��� �������� ���Ƿ� ū ������ ó��
                }
            }
            std::vector<int> con_btw_row_lines(H);
            for (int h = 0; h < H; h++) {
                int relate_cnt = 0;
                if (row_zeros[h] > 1) {
                    for (int w = 0; w < W; w++) {
                        if (col_zeros[w] > 1) {
                            if (costM[h * W + w] == 0) {
                                relate_cnt++;
                            }
                        }
                    }
                    con_btw_row_lines[h] = relate_cnt;
                }
                else {
                    con_btw_row_lines[h] = INT_MAX; // 0�� �ϳ��� ���� ��� �������� ���Ƿ� ū ������ ó��
                }
            }

            // �ּ� �������� ���� ���� Ž���Ͽ� ���� ó��
            float re_Col_value = *min_element(con_btw_col_lines.begin(), con_btw_col_lines.end());
            float re_Row_value = *min_element(con_btw_row_lines.begin(), con_btw_row_lines.end());
            if (re_Col_value < re_Row_value) {
                int re_Col_index = min_element(con_btw_col_lines.begin(), con_btw_col_lines.end()) - con_btw_col_lines.begin();
                coveredCols[re_Col_index] = true;
            }
            else {
                int re_Row_index = min_element(con_btw_row_lines.begin(), con_btw_row_lines.end()) - con_btw_row_lines.begin();
                coveredRows[re_Row_index] = true;
            }
        }
        step3_0(costM, coveredCols, coveredRows, H, W);
    }
}

// Step 3. 0�� ���� ���� �����ϴ� �� �Ǵ� ���� �ּ� ���� Ž��
void step3(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W) 
{
    //std::cout << "step3" << std::endl;
    std::fill(coveredRows.begin(), coveredRows.end(), 0); // ���� �ʱ�ȭ
    std::fill(coveredCols.begin(), coveredCols.end(), 0); // ���� �ʱ�ȭ

    step3_0(costM, coveredCols, coveredRows, H, W);
}

// step 4. �ּҼ� ��
// ã�� �ּҼ� ������ ��� ���ο� ���� �� ���� ���̿� ���ٸ�   -> step 6 ���� ���� �ĺ��� Ž��
// ã�� �ּҼ� ������ ��� ���ο� ���� �� ���� ���̺��� �۴ٸ� -> step 5 ��ó�� �۾�
void step4(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, int H, int W, float maxV, std::vector<std::vector<float>> &assign) 
{
    //std::cout << "step4" << std::endl;
    // ������� ã�� �� ���� ī��Ʈ
    int line_count = 0;
    for (int h = 0; h < coveredCols.size(); h++) {
        if (coveredCols[h]) line_count++;
    }
    for (int h = 0; h < coveredRows.size(); h++) {
        if (coveredRows[h]) line_count++;
    }

    if (std::min(H, W) == line_count){
        //std::cout << "step6" << std::endl;
        std::vector<std::vector<float>> map(H);
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                if (costM[h*W + w] == 0) {
                    map[h].push_back(w);
                }
            }
        }
        std::vector<int> check(W);
        std::vector<float> candidates;
        step6(H, 0, map, candidates, check, assign);
    }
    else {
        step5(costM, coveredCols, coveredRows, H, W, maxV, assign);
    }
}

// step 5. ��ó�� �۾�
// ������� ã�� ������ ������ �ʴ� ���� �� �ּڰ� Ž��
// ������ �ʴ� ���鿡 ���ؼ� �ּڰ����� ���� ó�� �� ���� �������� ������ �κп� ���ؼ��� �ּڰ��� ���� ó��
void step5(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, int H, int W, float maxV, std::vector<std::vector<float>> &assign) 
{
    //std::cout << "step5" << std::endl;
    float minV2 = maxV;
    for (int h = 0; h < H; h++) {
        if (coveredRows[h] == 0){
            for (int w = 0; w < W; w++) {
                if (coveredCols[w] == 0) {
                    if (minV2 > costM[h * W + w]) {
                        minV2 = costM[h * W + w];
                    }
                }
            }
        }
    }

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            if (coveredCols[w] == 0 && coveredRows[h] == 0) {
                costM[h * W + w] -= minV2;
            }
            if (coveredCols[w] == 1 && coveredRows[h] == 1) {
                costM[h * W + w] += minV2;
            }
        }
    }

    // �ٽ� �ּҼ� Ž��
    step3(costM, coveredCols, coveredRows, H, W);
    // �ٽ� �ּҼ� ��
    step4(costM, coveredCols, coveredRows, H, W, maxV, assign);
}

// step 6. ���� ���� �ĺ��� Ž��
void step6(int depth, int current, std::vector<std::vector<float>>& map, std::vector<float>& candidates, std::vector<int>& check, std::vector<std::vector<float>>& assign)
{
    if (depth == current) {
        assign.push_back(candidates);
    }
    else {
        for (int i = 0; i < map[current].size(); i++) {
            if (check[map[current][i]] == 0) {
                check[map[current][i]] = 1;
                candidates.push_back(map[current][i]);
                current++;
                step6(depth, current, map, candidates, check, assign);
                candidates.pop_back();
                current--;
                check[map[current][i]] = 0;
            }
        }
    }
}