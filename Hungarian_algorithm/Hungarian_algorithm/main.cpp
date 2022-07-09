#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip> // for setw()

// 콘솔창에 행렬값을 출력
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
// 콘솔창에 행렬값을 출력
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
// 데이터 파일로 저장
template <typename T>
void tofile(T* Buffer, int data_count, std::string fname = "../valid_py/from_C") 
{
    std::ofstream fs(fname, std::ios::binary);
    if (fs.is_open())
        fs.write((const char*)Buffer, data_count * sizeof(T));
    fs.close();
    std::cout << "[INFO] Done File Production to " << fname << std::endl;
}

// 비용 행렬 생성 
void gen_matrix(std::vector<std::vector<float>> &costMatrix, int& N, int& M, bool generate_costMatrix);
// Step 1. 각 행에서 최솟값을 구한 후 뺄셈 처리
void step1(std::vector<float> &costM, const int H, const int W);
// Step 2. 각 열에서 최솟값을 구한 후 뺄셈 처리
void step2(std::vector<float> &costM, const int H, const int W);
// Step 3. 0을 가장 많이 포함하는 행 또는 열을 최소 개수 탐색
void step3_0(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W);
void step3(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W);
// step 4. 최소선 평가
void step4(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, int H, int W, float maxV, std::vector<std::vector<float>>&assign);
// step 5. 재처리 작업
void step5(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, int H, int W, float maxV, std::vector<std::vector<float>>&assign);
// step 6. 최적 조합 후보군 탐색
void step6(int d, int c, std::vector<std::vector<float>>& map, std::vector<float>& candi, std::vector<int>& check, std::vector<std::vector<float>>& out);
// 헝가리안 알고리즘 실행 (문제에서 제공된 함수 정의 사용)
float Solve(const float **Cost, const int N, const int M, const int MODE, float* assignment_index);

int main() 
{
    //입력 준비 [N, M] OR [H, W] OR [rows, cols]
    int N = 5;                          // 인원수, ROWS, 행
    int M = 5;                          // 작업수, COLS, 열
    bool generate_costMatrix = false;   // 비용 행렬 생성 (true : [N, M] 행렬 랜덤 생성, false : TEST 예시 행렬 사용)
    bool valid_py = false;              // 검증 스크립트 실행 (true : 파이썬 검증 코드 실행)

    std::vector<std::vector<float>> costMatrix;         // 비용 행렬
    gen_matrix(costMatrix, N, M, generate_costMatrix);  // 비용 행렬 생성
    print_matrix(costMatrix, N, M, "Cost Matrix");      // 비용 행렬 출력

    const float **Cost = new const float*[N];
    for (int i = 0; i < N; i++){
        Cost[i] = costMatrix[i].data();
    }

    float *assignment_index_0 = new float[N];
    float *assignment_index_1 = new float[N];

    Solve(Cost, N, M, 0, assignment_index_0);
    Solve(Cost, N, M, 1, assignment_index_1);

    // python 검증 스크립트 수행
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

    // 자원(메모리) 해제
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

    // 만약 비용 행렬에 마이너스 값이 있으면, 최소값의 절대값으로 모두 더함 
    if (minV < 0) {
        for (int i = 0; i < costM.size(); i++) {
            costM[i] = fabs(maxV) + costM[i];
        }
        //print_matrix(costM, H, W, "cost Matrix (remove negative values)");
    }
    
    // 만약 최대 비용 모드 이면, 비용 행렬의 최대값에서 요소값을 뺀다.
    if (MODE == 1) { // MODE == 1 최대화, MODE == 0 최소화
        for (int i = 0; i < costM.size(); i++) {
            costM[i] = maxV - costM[i];
        }
        //print_matrix(costM, H, W, "cost Matrix (change for maximize)");
    }

    //Step 1. 각 행에서 최솟값을 구한 후 뺄셈 처리
    step1(costM, H, W); // O(N^2)

    //Step 2. 각 열에서 최솟값을 구한 후 뺄셈 처리
    step2(costM, H, W); // O(N^2)

    // Step 3. 행렬에서 0을 덮는 최소선 개수 탐색
    std::vector<int> coveredCols(W);
    std::vector<int> coveredRows(H);
    step3(costM, coveredCols, coveredRows, H, W); // O(N^4)

    // step 4
    // 찾은 최소선 개수가 행렬 가로와 세로 중 작은 길이와 같다면   -> step 6 최적 조합 후보군 탐색(마지막 step)
    // 찾은 최소선 개수가 행렬 가로와 세로 중 작은 길이보다 작다면 -> step 5 재처리 작업 -> step 3 -> step 4
    std::vector<std::vector<float>> candidates;// candidatase
    step4(costM, coveredCols, coveredRows, H, W, maxV, candidates);

    if (candidates.size() != 0) {
        // 후보 조합들의 비용 계산
        std::vector<float> candidates_cost;
        for (int c = 0; c < candidates.size(); c++) {
            float sum = 0;
            for (int h = 0; h < H; h++) {
                sum += Cost[h][int(candidates[c][h])];
            }
            candidates_cost.push_back(sum);
        }

        // MODE에 따라 최적 결과 도출
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

        // 결과 출력
        std::cout << "Mode " << MODE << " 인 경우 Total cost : " << cost_value;
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
        std::cout << "[ERROR] Something wrong... " << std::endl; // 오류시 예외 처리
    }

    return 0;
}

// 비용 행렬 생성 
void gen_matrix(std::vector<std::vector<float>> &costMatrix, int& N, int& M, bool generate_costMatrix) 
{
    if (generate_costMatrix) { // 랜덤 값 생성
        srand(time(0));
        for (int h = 0; h < N; h++) {
            std::vector<float> temp;
            for (int w = 0; w < M; w++) {
                temp.push_back(rand() % 100);
            }
            costMatrix.push_back(temp);
        }
    }
    else {// TEST 예시 행렬
        N = 3;
        M = 4;
        costMatrix = { {3, 7, 5, 11},
                       {5, 4, 6, 3},
                       {6, 10, 1, 1} };
    }
}

// Step 1. 각 행에서 최솟값을 구한 후 뺄셈 처리
void step1(std::vector<float> &costM, const int H, const int W) 
{
    //std::cout << "step1" << std::endl;
    for (int h = 0; h < H; h++) {
        // 1-1 각 행마다 최솟값 구하기
        float minV = costM[h * W];
        for (int w = 1; w < W; w++) {
            if (minV == 0) break;
            if (minV > costM[h * W + w]) {
                minV = costM[h * W + w];
            }
        }
        if (minV == 0) continue;
        // 1-2 각 행의 최솟값으로 행의 요소값 빼기
        for (int w = 0; w < W; w++) {
            costM[h * W + w] = costM[h * W + w] - minV;
        }
    }
}

// Step 2. 각 열에서 최솟값을 구한 후 뺄셈 처리
void step2(std::vector<float> &costM, const int H, const int W) 
{
    //std::cout << "step2" << std::endl;
    for (int w = 0; w < W; w++) {
        // 2-1 각 열마다 최솟값 구하기
        float minV = costM[w];
        for (int h = 1; h < H; h++) {
            if (minV == 0) break;
            if (minV > costM[h * W + w]) {
                minV = costM[h * W + w];
            }
        }
        if (minV == 0) continue;
        // 2-2 각 열의 최솟값으로 열의 요소값 빼기
        for (int h = 0; h < H; h++) {
            costM[h * W + w] = costM[h * W + w] - minV;
        }
    }
}

// Step 3-0. 최소 탐색
void step3_0(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W) 
{
    // 현재까지 찾은 선으로 덮이지 않는 영의 총 개수와 행과 열 방향에서 개수 탐색 
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
    if (zero_count == 0) { // 현재 남아 있는 영이 없다면, 종료
        // exit
    }
    else {
        int over_one_count = 0; // 영이 2개 이상 포함된 라인 수
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

        if (over_one_count == 0) { // 영을 1개만 갖는 라인들만 남을 경우
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
            // 영을 2개 이상 갖는 라인들간의 영에 대한 최소 연관성을 갖는 라인을 탐색하여 우선 선택
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
                    con_btw_col_lines[w] = INT_MAX; // 0이 하나인 선의 경우 연관성을 임의로 큰 값으로 처리
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
                    con_btw_row_lines[h] = INT_MAX; // 0이 하나인 선의 경우 연관성을 임의로 큰 값으로 처리
                }
            }

            // 최소 연관성을 갖는 라인 탐색하여 선택 처리
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

// Step 3. 0을 가장 많이 포함하는 행 또는 열을 최소 개수 탐색
void step3(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W) 
{
    //std::cout << "step3" << std::endl;
    std::fill(coveredRows.begin(), coveredRows.end(), 0); // 벡터 초기화
    std::fill(coveredCols.begin(), coveredCols.end(), 0); // 벡터 초기화

    step3_0(costM, coveredCols, coveredRows, H, W);
}

// step 4. 최소선 평가
// 찾은 최소선 개수가 행렬 가로와 세로 중 작은 길이와 같다면   -> step 6 최적 조합 후보군 탐색
// 찾은 최소선 개수가 행렬 가로와 세로 중 작은 길이보다 작다면 -> step 5 재처리 작업
void step4(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, int H, int W, float maxV, std::vector<std::vector<float>> &assign) 
{
    //std::cout << "step4" << std::endl;
    // 현재까지 찾은 선 개수 카운트
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

// step 5. 재처리 작업
// 현재까지 찾은 선으로 덮이지 않는 값들 중 최솟값 탐색
// 덮이지 않는 값들에 대해서 최솟값으로 뺄셈 처리 및 선이 이중으로 겹쳐진 부분에 대해서는 최솟값을 덧셈 처리
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

    // 다시 최소선 탐색
    step3(costM, coveredCols, coveredRows, H, W);
    // 다시 최소선 평가
    step4(costM, coveredCols, coveredRows, H, W, maxV, assign);
}

// step 6. 최적 조합 후보군 탐색
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