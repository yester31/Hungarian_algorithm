#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
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
void tofile(T* Buffer, int data_count, std::string fname = "../valid_py/from_C") {
    std::ofstream fs(fname, std::ios::binary);
    if (fs.is_open())
        fs.write((const char*)Buffer, data_count * sizeof(T));
    fs.close();
    std::cout << "Done! file production to " << fname << std::endl;
}

void step1(std::vector<float> &costM, const int H, const int W);
void step2(std::vector<float> &costM, const int H, const int W);
void step3(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W);
void step4(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, int H, int W, float maxV, std::vector<std::vector<float>>&assign);
void step5(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, int H, int W, float maxV, std::vector<std::vector<float>>&assign);
void step6(int d, int c, std::vector<std::vector<float>>& map, std::vector<float>& candi, std::vector<int>& check, std::vector<std::vector<float>>& out);
float Solve(const float **Cost, const int N, const int M, const int MODE, float* assignment_index);
void gen_matrix(std::vector<std::vector<float>> &costMatrix, int& N, int& M, bool generate_costMatrix);

int main() {
    //Step 0. 입력 준비
    // [N, M] OR [H, W] OR [rows, cols]
    int N = 5; // 인원 수, ROWS, 행 수, 
    int M = 5; // 작업 수, COLS, 열 수, 
    int generate_costMatrix = false; // true : 랜덤 생성, false : TEST 예시 행렬 사용
    bool valid_py = true;           // true : 파이썬 검증 코드 실행

    std::vector<std::vector<float>> costMatrix;
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

    if (valid_py) {
        // python 검증 스크립트 수행
        std::cout << "\n *Validation with python" << std::endl;
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
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            float elt = Cost[h][w];
            costM.push_back(elt);
            if (maxV < elt) maxV = elt;
        }
    }
    // 만약 최대 비용 모드 이면, 비용 행렬의 최대값에 요소값을 뺀다.
    if (MODE == 1) { // MODE == 1 최대화, MODE == 0 최소화
        for (int i = 0; i < costM.size(); i++) {
            costM[i] = maxV - costM[i];
        }
    }
    //print_matrix(costM, H, W, "cost Matrix 0");

    //Step 1. 각 행에서 최솟값을 구한 후 뺄셈 처리
    step1(costM, H, W);
    //Step 2. 각 열에서 최솟값을 구한 후 뺄셈 처리
    step2(costM, H, W);
    std::vector<int> coveredCols(W);
    std::vector<int> coveredRows(H);
    // Step 3. 0을 가장 많이 포함하는 행 또는 열을 최소 개수 탐색
    step3(costM, coveredCols, coveredRows, H, W);
    std::vector<std::vector<float>> out;
    // step 4
    step4(costM, coveredCols, coveredRows, H, W, maxV, out);

    if (out.size() != 0) {
        std::vector<float> candi_cost;
        for (int i = 0; i < out.size(); i++) { // candidates
            float sum = 0;
            for (int h = 0; h < H; h++) {
                sum += Cost[h][int(out[i][h])];
            }
            candi_cost.push_back(sum);
        }

        float cost_value;
        int cost_index;
        if (MODE == 1) {
            cost_value = *max_element(candi_cost.begin(), candi_cost.end());
            cost_index = max_element(candi_cost.begin(), candi_cost.end()) - candi_cost.begin();
        }
        else {
            cost_value = *min_element(candi_cost.begin(), candi_cost.end());
            cost_index = min_element(candi_cost.begin(), candi_cost.end()) - candi_cost.begin();
        }

        for (int i = 0; i < H; i++) {
            assignment_index[i] = out[cost_index][i];
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

void gen_matrix(std::vector<std::vector<float>> &costMatrix, int& N, int& M, bool generate_costMatrix) {
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
    else {// 예시 행렬
        N = 3;
        M = 4;
        costMatrix = { {3, 7, 5, 11},
                       {5, 4, 6, 3},
                       {6, 10, 1, 1} };
    }
}


//Step 1. 각 행에서 최솟값을 구한 후 뺄셈 처리
void step1(std::vector<float> &costM, const int H, const int W) {
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
    //print_matrix(costM, H, W, "Step 1");
    //std::cout << "=========================" << std::endl;
}

//Step 2. 각 열에서 최솟값을 구한 후 뺄셈 처리
void step2(std::vector<float> &costM, const int H, const int W) {
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
    //print_matrix(costM, H, W, "Step 2");
    //std::cout << "=========================" << std::endl;
}

void step3_4(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W) {
    int zero_count = 0;
    std::vector<int> Col_zeros(W);
    std::vector<int> Row_zeros(H);
    for (int h = 0; h < H; h++) {
        if (coveredRows[h] == 0) {
            for (int w = 0; w < W; w++) {
                if (coveredCols[w] == 0) {
                    if (costM[h * W + w] == 0) {
                        Col_zeros[w]++;
                        Row_zeros[h]++;
                        zero_count++;
                    }
                }
            }
        }
    }
    if (zero_count == 0) { // 현재 남아 있는 영이 없다면, 종료
        //exit()
    }
    else {
        int over_one_count = 0; // 라인들 중 1개 이상 라인 수
        for (int h = 0; h < H; h++) {
            if (Row_zeros[h] > 1) { 
                over_one_count++;
            }
        }
        for (int w = 0; w < W; w++) {
            if (Col_zeros[w] > 1) {
                over_one_count++;
            }
        }

        float Col_value = *max_element(Col_zeros.begin(), Col_zeros.end());
        float Row_value = *max_element(Row_zeros.begin(), Row_zeros.end());
        if (over_one_count == 0) { // 영을 1개만 갖는 라인들만 남을 경우
            if (Col_value > Row_value) {
                int Col_index = max_element(Col_zeros.begin(), Col_zeros.end()) - Col_zeros.begin();
                coveredCols[Col_index] = true;
            }
            else {
                int Row_index = max_element(Row_zeros.begin(), Row_zeros.end()) - Row_zeros.begin();
                coveredRows[Row_index] = true;
            }
        }
        else {
            // 영을 1개 이상 갖는 라인들 간의 영에 대한 최소 연관성을 갖는 라인 탐색하여 우선 선택
            std::vector<int> re_Col_zeros(W);
            for (int w = 0; w < W; w++) {
                int relate_cnt = 0;
                if (Col_zeros[w] > 1) {
                    for (int h = 0; h < H; h++) {
                        if (Row_zeros[h] > 1) {
                            if (costM[h * W + w] == 0) {
                                relate_cnt++;
                            }
                        }
                    }
                    re_Col_zeros[w] = relate_cnt;
                }
                else {
                    re_Col_zeros[w] = 999999;
                }
            }

            std::vector<int> re_Row_zeros(H);
            for (int h = 0; h < H; h++) {
                int relate_cnt = 0;
                if (Row_zeros[h] > 1) {
                    for (int w = 0; w < W; w++) {
                        if (Col_zeros[w] > 1) {
                            if (costM[h * W + w] == 0) {
                                relate_cnt++;
                            }
                        }
                    }
                    re_Row_zeros[h] = relate_cnt;
                }
                else {
                    re_Row_zeros[h] = 999999;
                }
            }
            float re_Col_value = *min_element(re_Col_zeros.begin(), re_Col_zeros.end());
            float re_Row_value = *min_element(re_Row_zeros.begin(), re_Row_zeros.end());
            float re_min = 0;

            if (re_Col_value < re_Row_value) {
                int re_Col_index = min_element(re_Col_zeros.begin(), re_Col_zeros.end()) - re_Col_zeros.begin();
                coveredCols[re_Col_index] = true;
            }
            else {
                int re_Row_index = min_element(re_Row_zeros.begin(), re_Row_zeros.end()) - re_Row_zeros.begin();
                coveredRows[re_Row_index] = true;
            }
        }
        step3_4(costM, coveredCols, coveredRows, H, W);
    }
}


void step3_dfs2(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W, 
    std::vector<std::pair<int, int>> &checkM, int depth, int& current, std::vector<int> &Col_check2, std::vector<int> &Row_check2) {
    int zero_count = 0;
    for (int h = 0; h < H; h++) {
        if (Row_check2[h] == 0) {
            for (int w = 0; w < W; w++) {
                if (Col_check2[w] == 0) {
                    if (costM[h * W + w] == 0) {
                        zero_count++;
                    }
                }
            }
        }
    }
    if(depth == current){ // 종료
        if (zero_count == 0) { 
            //exit() 찾음
            coveredCols.assign(Col_check2.begin(), Col_check2.end());
            coveredRows.assign(Row_check2.begin(), Row_check2.end());
        }
    }
    else {
        Row_check2[checkM[current].first] = 1;// h
        current++;
        step3_dfs2(costM, coveredCols, coveredRows, H, W, checkM, depth, current, Col_check2, Row_check2);
        current--;
        Row_check2[checkM[current].first] = 0;// h

        Col_check2[checkM[current].second] = 1;// w
        current++;
        step3_dfs2(costM, coveredCols, coveredRows, H, W, checkM, depth, current, Col_check2, Row_check2);
        current--;
        Col_check2[checkM[current].second] = 0;// w
    }

}

void step3_3(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W) {
    std::fill(coveredRows.begin(), coveredRows.end(), 0);
    std::fill(coveredCols.begin(), coveredCols.end(), 0);
    int depth = 0;
    int current = 0;
    std::vector<int> Col_check(W);
    std::vector<int> Row_check(H);
    std::vector<std::pair<int, int>> checkM;
    std::vector<float> checkM2(H*W);
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            if (Col_check[w] == 0 && Row_check[h] == 0) {
                if (costM[h * W + w] == 0) {
                    checkM2[h * W + w] = 1;
                    checkM.push_back({ h, w });
                    Col_check[w] = 1;
                    Row_check[h] = 1;
                    depth++;
                    break;
                }
            }
        }
    }
    print_matrix(checkM2, H, W, "checkM2");
    std::vector<int> Col_check2(W);
    std::vector<int> Row_check2(H);
    step3_dfs2(costM, coveredCols, coveredRows, H, W, checkM, depth, current, Col_check2, Row_check2);

}

void step3_2(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W) {
    std::fill(coveredRows.begin(), coveredRows.end(), 0);
    std::fill(coveredCols.begin(), coveredCols.end(), 0);
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            if (costM[h * W + w] == 0 && coveredCols[w] == 0) {
                coveredCols[w] = 1;
                break;
            }
        }
    }
    //print_matrix(coveredCols, 1, W, "coveredCols");
    for (int w = 0; w < W; w++) {
        if (coveredCols[w] == 0) {
            for (int h = 0; h < H; h++) {
                if (costM[h * W + w] == 0 && coveredRows[h] == 0) {
                    for (int starCol = 0; starCol < W; starCol++) {
                        if (w != starCol && costM[h * W + starCol] == 0) {
                            coveredCols[starCol] = false;
                            for (int row = 0; row < H; row++) {
                                if (h != row && costM[row * W + starCol] == 0) {
                                    coveredCols[starCol] = true;
                                    break;
                                }
                            }
                        }
                    }
                    coveredRows[h] = true;
                }
            }
        }
    }
    print_matrix(coveredCols, 1, W, "coveredCols");
    print_matrix(coveredRows, H, 1, "coveredRows");
}

void step3_1(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W) {
    int zero_count = 0;
    std::vector<int> Col_zeros(W);
    std::vector<int> Row_zeros(H);
    for (int h = 0; h < H; h++) {
        if (coveredRows[h] == 0) {
            for (int w = 0; w < W; w++) {
                if (coveredCols[w] == 0) {
                    if (costM[h * W + w] == 0) {
                        Col_zeros[w]++;
                        Row_zeros[h]++;
                        zero_count++;
                    }
                }
            }
        }
    }
    if (zero_count == 0) { // 현재 남아 있는 제로가 없다면, 종료
        //exit()
    }
    else {
        // 최대 제로 개수를 보유한 후보 라인 탐색
        float Col_value = *max_element(Col_zeros.begin(), Col_zeros.end());
        float Row_value = *max_element(Row_zeros.begin(), Row_zeros.end());
        float max = 0;
        if (Col_value > Row_value) { max = Col_value; }
        else { max = Row_value; }

        int max_count = 0;
        for (int h = 0; h < H; h++) {
            if (Row_zeros[h] == max) { max_count++; }
        }
        for (int w = 0; w < W; w++) {
            if (Col_zeros[w] == max) { max_count++; }
        }

        if (max_count == 1) { // 최대 제로 개수를 보유한 후보 라인이 하나 라면
            if (Col_value > Row_value) {
                int Col_index = max_element(Col_zeros.begin(), Col_zeros.end()) - Col_zeros.begin();
                coveredCols[Col_index] = true;
            }
            else {
                int Row_index = max_element(Row_zeros.begin(), Row_zeros.end()) - Row_zeros.begin();
                coveredRows[Row_index] = true;
            }
        }
        else {
            // 최대 제로 개수를 보유한 후보 라인이 복수라면, 그 라인들 간의 제로에 대한 최소 연관성을 갖는 라인 탐색
            std::vector<int> re_Col_zeros(W);
            for (int w = 0; w < W; w++) {
                int relate_cnt = 0;
                if (Col_zeros[w] == max) {
                    for (int h = 0; h < H; h++) {
                        if (Row_zeros[h] == max) {
                            if (costM[h * W + w] == 0) {
                                relate_cnt++;
                            }
                        }
                    }
                    re_Col_zeros[w] = relate_cnt;
                }
                else {
                    re_Col_zeros[w] = 999999;
                }
            }

            std::vector<int> re_Row_zeros(H);
            for (int h = 0; h < H; h++) {
                int relate_cnt = 0;
                if (Row_zeros[h] == max) {
                    for (int w = 0; w < W; w++) {
                        if (Col_zeros[w] == max) {
                            if (costM[h * W + w] == 0) {
                                relate_cnt++;
                            }
                        }
                    }
                    re_Row_zeros[h] = relate_cnt;
                }
                else {
                    re_Row_zeros[h] = 999999;
                }
            }
            float re_Col_value = *min_element(re_Col_zeros.begin(), re_Col_zeros.end());
            float re_Row_value = *min_element(re_Row_zeros.begin(), re_Row_zeros.end());
            float re_min = 0;
            if (re_Col_value < re_Row_value) {
                int re_Col_index = min_element(re_Col_zeros.begin(), re_Col_zeros.end()) - re_Col_zeros.begin();
                coveredCols[re_Col_index] = true;
            }
            else {
                int re_Row_index = min_element(re_Row_zeros.begin(), re_Row_zeros.end()) - re_Row_zeros.begin();
                coveredRows[re_Row_index] = true;
            }
        }
        step3_1(costM, coveredCols, coveredRows, H, W);
    }
}

void step3_0(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W) {
    int zero_count = 0;
    std::vector<int> Col_zeros(W);
    std::vector<int> Row_zeros(H);
    for (int h = 0; h < H; h++) {
        if (coveredRows[h] == 0) {
            for (int w = 0; w < W; w++) {
                if (coveredCols[w] == 0) {
                    if (costM[h * W + w] == 0) {
                        Col_zeros[w]++;
                        Row_zeros[h]++;
                        zero_count++;
                    }
                }
            }
        }
    }
    if (zero_count == 0) { // 현재 남아 있는 제로가 없다면, 종료
        //exit()
    }
    else {
        // 최대 제로 개수를 보유한 후보 라인 탐색
        float Col_value = *max_element(Col_zeros.begin(), Col_zeros.end());
        float Row_value = *max_element(Row_zeros.begin(), Row_zeros.end());
        float max = 0;
        if (Col_value > Row_value) { max = Col_value; }
        else { max = Row_value; }

        int max_count = 0;
        for (int h = 0; h < H; h++) {
            if (Row_zeros[h] == max) { max_count++; }
        }
        for (int w = 0; w < W; w++) {
            if (Col_zeros[w] == max) { max_count++; }
        }

        if (max_count == 1) { // 최대 제로 개수를 보유한 후보 라인이 하나 라면
            if (Col_value > Row_value) {
                int Col_index = max_element(Col_zeros.begin(), Col_zeros.end()) - Col_zeros.begin();
                coveredCols[Col_index] = true;
            }
            else {
                int Row_index = max_element(Row_zeros.begin(), Row_zeros.end()) - Row_zeros.begin();
                coveredRows[Row_index] = true;
            }
        }
        else {
            // 최대 제로 개수를 보유한 후보 라인이 복수라면, 그 라인들 간의 제로에 대한 최소 연관성을 갖는 라인 탐색
            std::vector<int> re_Col_zeros(W);
            for (int w = 0; w < W; w++) {
                int relate_cnt = 0;
                if (Col_zeros[w] == max) {
                    for (int h = 0; h < H; h++) {
                        if (Row_zeros[h] == max) {
                            if (costM[h * W + w] == 0) {
                                relate_cnt++;
                            }
                        }
                    }
                    re_Col_zeros[w] = relate_cnt;
                }
                else {
                    re_Col_zeros[w] = 999999;
                }
            }

            std::vector<int> re_Row_zeros(H);
            for (int h = 0; h < H; h++) {
                int relate_cnt = 0;
                if (Row_zeros[h] == max) {
                    for (int w = 0; w < W; w++) {
                        if (Col_zeros[w] == max) {
                            if (costM[h * W + w] == 0) {
                                relate_cnt++;
                            }
                        }
                    }
                    re_Row_zeros[h] = relate_cnt;
                }
                else {
                    re_Row_zeros[h] = 999999;
                }
            }
            float re_Col_value = *min_element(re_Col_zeros.begin(), re_Col_zeros.end());
            float re_Row_value = *min_element(re_Row_zeros.begin(), re_Row_zeros.end());
            float re_min = 0;
            if (re_Col_value < re_Row_value) {
                int re_Col_index = min_element(re_Col_zeros.begin(), re_Col_zeros.end()) - re_Col_zeros.begin();
                coveredCols[re_Col_index] = true;
            }
            else {
                int re_Row_index = min_element(re_Row_zeros.begin(), re_Row_zeros.end()) - re_Row_zeros.begin();
                coveredRows[re_Row_index] = true;
            }
        }
        step3_0(costM, coveredCols, coveredRows, H, W);
    }
}


// Step 3. 0을 가장 많이 포함하는 행 또는 열을 최소 개수 탐색
void step3(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, const int H, const int W) {
    //std::cout << "step3" << std::endl;
    //print_matrix(costM, H, W, "step3");
    std::fill(coveredRows.begin(), coveredRows.end(), 0);
    std::fill(coveredCols.begin(), coveredCols.end(), 0);

    //step3_1(costM, coveredCols, coveredRows, H, W);
    //step3_2(costM, coveredCols, coveredRows, H, W);
    //step3_3(costM, coveredCols, coveredRows, H, W);
    step3_4(costM, coveredCols, coveredRows, H, W);

    //print_matrix(coveredCols, 1, W, "coveredCols");
    //print_matrix(coveredRows, H, 1, "coveredRows");
    //std::cout << "=========================" << std::endl;
}

// step 6. 최적 조합 후보군 찾기
void step6(int d, int c, std::vector<std::vector<float>>& map, std::vector<float>& candi, std::vector<int>& check, std::vector<std::vector<float>>& out) {
    if (d == c) {
        out.push_back(candi);
    }
    else {
        for (int i = 0; i < map[c].size(); i++) {
            if (check[map[c][i]] == 0) {
                check[map[c][i]] = 1;
                candi.push_back(map[c][i]);
                c++;
                step6(d, c, map, candi, check, out);
                candi.pop_back();
                c--;
                check[map[c][i]] = 0;
            }
        }
    }
}

// step 4
// 찾은 선의 수가 최소 길이와 같다면 step 6
// 찾은 선의 수 최소 길이 보다 작다면 step 5
void step4(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, int H, int W, float maxV, std::vector<std::vector<float>> &assign) {
    //std::cout << "step4" << std::endl;
    int line_count = 0;
    for (int h = 0; h < coveredCols.size(); h++) {
        if (coveredCols[h]) line_count++;
    }
    for (int h = 0; h < coveredRows.size(); h++) {
        if (coveredRows[h]) line_count++;
    }

    if (std::min(H, W) == line_count){
        //std::cout << "step6" << std::endl;
        //print_matrix(coveredCols, 1, W, "coveredCols");
        //print_matrix(coveredRows, H, 1, "coveredRows");
        //print_matrix(costM, H, W, "Step 6");
        std::vector<std::vector<float>> map(H);
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                if (costM[h*W + w] == 0) {
                    map[h].push_back(w);
                }
            }
        }
        std::vector<int> check(W);
        std::vector<float> candi;
        step6(H, 0, map, candi, check, assign);
    }
    else {
        step5(costM, coveredCols, coveredRows, H, W, maxV, assign);
    }
    //std::cout << "=========================" << std::endl;
}

// 5-1) step 3. 에서 찾은 라인으로 covered 되지 않은 elements 중 최솟값을 찾는다.
// 5-2) 선에 덮혀지지 않은 값들에 대해서 뺄셈 처리 및 선이 이중으로 겹쳐진 부분에 대해 덧셈 처리
void step5(std::vector<float> &costM, std::vector<int> &coveredCols, std::vector<int> &coveredRows, int H, int W, float maxV, std::vector<std::vector<float>> &assign) {
    //std::cout << "step5" << std::endl;
    float minV2 = maxV;
    for (int h = 0; h < H; h++) {
        if (coveredRows[h] == 0)
            for (int w = 0; w < W; w++) {
                if (coveredCols[w] == 0) {
                    if (minV2 > costM[h * W + w]) {
                        minV2 = costM[h * W + w];
                    }
                }
            }
    }
    //std::cout << "minV2 : " << minV2 << std::endl;
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
    //print_matrix(costM, H, W, "Step 5");
    step3(costM, coveredCols, coveredRows, H, W);
    step4(costM, coveredCols, coveredRows, H, W, maxV, assign);
    //std::cout << "=========================" << std::endl;
}
