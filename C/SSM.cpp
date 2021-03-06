//
// Created by ziga on 5.7.2017.
//

#include "SSM.h"
#include "zStrzinar.h"
#include "utility.h"

using namespace cv;
using namespace std;

void run_SSM(Colorspace colorSpace,
             cv::Size sizeMask,
             bool use_uniform_component,
             std::string type_of_em,
             int maxEMsteps,
             vector<cv::Mat> &mix_w,
             cv::Mat &PI_i,
             cv::Mat &data,
             vector<cv::Mat> &mix_Mu,
             vector<cv::Mat> &mix_Cov,
             vector<cv::Mat> prior_mix_Mu,
             vector<cv::Mat> prior_mix_Prec,
             bool use_prior_on_mixture,
             long double epsilon,
             cv::Mat &Q_sum_large,
             cv::Mat &mix_PI_i){

    double p_unknown = GetUnknownWeightForTheFeatureModel(colorSpace, sizeMask, use_uniform_component);
    bool use_gauss = false;
    int sizMix = (int)mix_w.size();

    // construct convolution kernels H_0 and H_1 for the em posteriors
    cv::Mat H_0, H_1;
    GetConvolutionKernel(type_of_em, sizeMask, H_0, H_1);
    if (PI_i.empty())
    {
        double estimate = 1.0/sizMix - p_unknown/sizMix;
        PI_i = cv::Mat_<cv::Vec4d>(sizeMask.height, sizeMask.width, cv::Vec4d(estimate, estimate, estimate, p_unknown));
    }

    cv::Mat PI_i0 = PI_i.clone();
    int counter = maxEMsteps;
    cv::Mat p(sizMix+1,data.cols,CV_64F, 0.0);

    //initialize empty precision matrix
    cv::Mat prec = cv::Mat();
    cv::Mat Q_sum;
    while (counter > 1)
    {
        for (int i = 0; i < sizMix; i++)
        {
            if (type_of_em.compare("em_seg") == 0)
            {
                cv::Mat pdf = normpdf(data.clone(), mix_Mu[i], prec, mix_Cov[i], (double) epsilon);
                pdf = pdf.t();
                pdf.copyTo(p.row(i));
            }
            else
            {
                cv::Mat pdf = normpdf(data.clone(), mix_Mu[i], prec, mix_Cov[i], (double) epsilon);
                pdf = pdf.t()*mix_w[i];
                pdf.row(0).copyTo(p.row(i));
            }
        }

        cv::Mat row = p.row(sizMix);
        row = row*0 + p_unknown;
        row.copyTo(p.row(sizMix));
        cv::Mat P_i(PI_i.rows, PI_i.cols, CV_64FC4);

        //for the paper
        p = p.t();
        p = p.reshape(sizMix+1, sizeMask.height).t();

        P_i = PI_i.mul(p) + epsilon;
        if (!use_uniform_component)
        {
            for (int i = 0; i < P_i.rows; i++)
            {
                for (int j = 0; j < P_i.cols; j++)
                {
                    P_i.at<cv::Vec4f>(j,i)[3] = 0;
                }
            }
        }

        //prepare the input for bsxfun
        int rows = P_i.rows;
        cv::Mat bsxParam;
        bsxParam = P_i.reshape(1, rows*P_i.cols);
        cv::reduce(bsxParam, bsxParam, 1, CV_REDUCE_SUM);
        bsxParam = bsxParam.reshape(1,rows);
        //cv::divide(1.0, bsxParam, bsxParam);
        P_i = Bsxfun(P_i, bsxParam);

        cv::Mat S_i;
        cv::flip(H_0,H_0, -1);//switch order in both axes
        cv::filter2D(PI_i, S_i, PI_i.depth(), H_0,cv::Point(-1,-1), 0.0,cv::BORDER_REPLICATE); // Filtrira PI_i z jedrom H_0, rezultat je S_i
        S_i = PI_i.mul(S_i);

        //prepare the input for bsxfun
        rows = S_i.rows;
        bsxParam = S_i.reshape(1, rows*S_i.cols);
        cv::reduce(bsxParam, bsxParam, 1, CV_REDUCE_SUM);
        bsxParam = bsxParam.reshape(1,rows);
        S_i = Bsxfun(S_i, bsxParam);

        cv::Mat Q_i;
        cv::filter2D(P_i, Q_i, P_i.depth(), H_0,cv::Point(-1,-1), 0.0,cv::BORDER_REPLICATE);
        Q_i = P_i.mul(Q_i);
        //prepare the input for bsxfun
        rows = Q_i.rows;
        bsxParam = Q_i.reshape(1, rows*Q_i.cols);
        cv::reduce(bsxParam, bsxParam, 1, CV_REDUCE_SUM);
        bsxParam = bsxParam.reshape(1,rows);
        Q_i = Bsxfun(Q_i, bsxParam);
        cv::Mat S_sum;
        cv::flip(H_1,H_1, -1);//switch order in both axes
        cv::filter2D(Q_i, Q_sum, Q_i.depth(), H_1,cv::Point(-1,-1), 0.0,cv::BORDER_REPLICATE);
        cv::filter2D(S_i, S_sum, S_i.depth(), H_1,cv::Point(-1,-1), 0.0,cv::BORDER_REPLICATE);
        cv::add(Q_sum, S_sum, PI_i);
        PI_i = PI_i*0.25;

        //TODO: can reshape and set 4th column to 0 and reshape back
        if (!use_uniform_component)
        {
            for (int i = 0; i < PI_i.rows; i++)
            {
                for (int j = 0; j < PI_i.cols; j++)
                {
                    PI_i.at<cv::Vec4f>(j,i)[3] = 0;
                }
            }
        }

        cv::Mat Q_sumT;
        Q_sumT = Q_sum.t(); // prej je bilo tukaj Q_sum = Q_sum.t(); ???
        p = Q_sumT.reshape(1,sizeMask.area()).t(); // Q_sumT prej sploh ni bil inicializiran!

        //measure the change in posterior distribution
        cv::Mat d_pi, tmpMat, tmpMat2;
        cv::sqrt(PI_i0, tmpMat);
        cv::sqrt(PI_i, tmpMat2);
        d_pi = tmpMat - tmpMat2;
        d_pi = d_pi.reshape(1, d_pi.rows*d_pi.cols);
        cv::reduce(d_pi, d_pi, 1, CV_REDUCE_SUM);
        //d_pi = d_pi.reshape(1, rows);

        cv::sort(d_pi,d_pi, CV_SORT_EVERY_COLUMN);
        double max;
        double min;
        cv::minMaxLoc(d_pi, &min,&max);
        int mid = (int)std::ceil(d_pi.rows/2);
        cv::Mat dpi = d_pi(cv::Range(mid,d_pi.rows),cv::Range::all());
        cv::Scalar loglik_new = cv::mean(dpi);
        if (loglik_new.val[0] > 0.0008) // 0.0008 original: 0.01
        {
            PI_i0 = PI_i;
        }
        else
        {
            break;
        }

        cv::Mat lp;
        cv::reduce(p,lp,0,CV_REDUCE_SUM);

        cv::repeat(lp, 4,1,lp);
        p = Bsxfun(p, lp);
        CvScalar x = cv::sum(PI_i);
        cv::Matx14d a_i(x.val[0], x.val[1], x.val[2], x.val[3]); //cv::Mat(1,4,CV_64F,x);
        cv::Mat alpha_i = cv::Mat(a_i);
        CvScalar sum = cv::sum(alpha_i);
        alpha_i = alpha_i/sum.val[0];

        cv::Mat w_data;
        cv::Mat x_mu;
        cv::Mat x_2_mu;
        cv::Mat c;
        // Na tej točki so matrike mix_Cov[] enake kot v Matlabu
        for (int k = 0; k < sizMix; k++)
        {

            cv::Mat pk = p.row(k);
            cv::Mat rep_pk = cv::repeat(pk, 5, 1);
            double sum_pk = cv::sum(pk).val[0];

            w_data = Bsxfun(data,rep_pk,TIMES);

            cv::reduce(w_data, x_mu, 1, CV_REDUCE_SUM);
            x_mu = x_mu/sum_pk;
            x_2_mu = (data*w_data.t())/sum_pk ;
            c = x_2_mu - x_mu*x_mu.t();
            // naive update of mean and covariance
            mix_Cov[k] = c.clone();
            x_mu.copyTo(mix_Mu[k]);
            mix_w[k].at<double>(0) = alpha_i.at<double>(0,k);
            cv::Mat res;
            if (use_prior_on_mixture)
            {
                int i_c_d=k;
                int i_c_0 = k;

                res = mergePd(mix_Mu[i_c_d], mix_Cov[k],
                              prior_mix_Mu[i_c_0],
                              cv::Mat(), prior_mix_Prec[i_c_0]);
                res.copyTo(mix_Mu[i_c_d]);
            }
        }
        // Na tej točki je ujemanje matrik mix_Cov[] z Matlabom že dosti slabše

        cv::Mat mix_w_sum_mat(1,1,CV_64F,Scalar(0));
        double mix_w_sum;
        int i;
        for (i=0; i<mix_w.size(); i++){
            mix_w_sum_mat += mix_w[i];
        }
        mix_w_sum = mix_w_sum_mat.at<double>(0);
        for (i=0; i<mix_w.size(); i++){
            mix_w[0] = mix_w[0]/mix_w_sum;
        }
        counter = counter-1;
    }


    Q_sum_large = Q_sum;
    mix_PI_i = Q_sum; //PI_i ;
}

Colorspace ResolveColorspace(std::string color){
/**
Resolves and returns enum for a given color space

@param color should be in lowercase
*/
    if(color.compare("hsv") == 0){
        return HSV;
    }else if(color.compare("rgb") == 0){
        return RGB;
    }else if(color.compare("ycrcb") == 0){
        return YCRCB;
    }else if(color.compare("lab") == 0){
        return LAB;
    }else if(color.compare("ycrs") == 0){
        return YCRS;
    }
    else{
        return NONE;
    }
}

double GetUnknownWeightForTheFeatureModel(Colorspace type_colorspace,cv::Size sizeMask, bool use_uniform_component){
    double p_unknown = 0;
    if (!use_uniform_component){
        return 0;
    }

    double p = sizeMask.area();
    switch(type_colorspace){
        case HSV:
            p_unknown = 1/p ;
            break;
        case RGB:
            p_unknown = 1/(p * cv::pow(255.0,3.0)) * 0.001;
            break;
        case YCRCB:
            p_unknown = 0.001*1/(p * 10988544) ;
            break;
        case LAB:
            p_unknown = cv::pow(10.0,-2.0) *  1 / (p * cv::pow(255.0,3.0)) ;
            break;
        case YCRS:
            p_unknown = 0.01*1/(p * 49056) ;
            break;
    }
    return p_unknown;
}

void  GetConvolutionKernel(std::string type_of_em, cv::Size sizeMask, cv::Mat& H_0, cv::Mat& H_1){
    /**
    Calculates gaussian kernels used in segmentation

    @param type_of_em (ie: em_seg)
    @param sizeMask size of a mask, usually a vector of 2 doubles (dimensions)
    @param H_0 pointer to H_0
    @param H_1 pointer to H_1
    */
    double scale_filter = 0.1;
    if (type_of_em.compare("em_seg") == 0)
    {
        double hsize = std::ceil(0.2*(sizeMask.height*scale_filter - 1));
        cv::Mat H_0_vector = cv::getGaussianKernel(hsize*2+1, hsize/1.5);
        H_0 =  H_0_vector*H_0_vector.t();
        int center = hsize;

        //set anchor element to 0
        H_0.at<double>(center, center) = 0;
        //and normalize H_0
        cv::Scalar sum = cv::sum(H_0);
        H_0 = (1/sum(0))*H_0;
        H_0.copyTo(H_1);
        H_1.at<double>(center, center) = 1;
    }
    else if(type_of_em.compare("em_basic") == 0){
        int hsize = std::ceil(0.2*(sizeMask.height*scale_filter - 1));
        cv::Mat H_0_vector = cv::getGaussianKernel(hsize*2+1, hsize/1.5);
        H_0 = H_0_vector.t()*H_0_vector;
    }
}

cv::Mat normpdf(cv::Mat x, cv::Mat mu, cv::Mat prec, cv::Mat sigma, double epsilon){
    cv::Mat A;
    cv::Mat repeatedX;
    double prodEigenVals = 1.0;
    //generate eigenvalue matrix 5�5 populated with zeros
    cv::Mat S = cv::Mat(5,5, CV_64F, double(0));
    cv::Mat S_tmp = cv::Mat(5,5, CV_64F, double(0));

    if (prec.empty())
    {
        cv::SVD svd(sigma);
        cv::Mat s(svd.w.rows, svd.w.cols, CV_64F);
        cv::Mat s_tmp(svd.w.rows, svd.w.cols, CV_64F);
        svd.w.copyTo(s);
        for (int i = 0; i < svd.w.rows; i++)
        {
            if(svd.w.at<double>(i,0) < epsilon)
            {
                svd.w.at<double>(i,0) = 1;
            }
            //svd.w.at<double>(i,0) = 1/sqrt(svd.w.at<double>(i,0));
            prodEigenVals *= svd.w.at<double>(i,0);
        }

        //set s as diagonal in it
        S = S.diag(s);
        cv::sqrt(s,s_tmp);
        s_tmp = 1/s_tmp;
        S_tmp = S_tmp.diag(s_tmp);

        repeatedX = cv::repeat(mu, 1, x.cols);
        cv::subtract(x, repeatedX, x); //(x-mu)
        A = (x).t()*(svd.u*S_tmp); //x'*(U*s));

        ~S;
    }
    else
    {
        cv::SVD svd(prec);
        for (int i = 0; i < svd.w.rows; i++)
        {
            svd.w.at<double>(i,0) = sqrt(svd.w.at<double>(i,0));
        }

        repeatedX = cv::repeat(mu, 1, x.cols);
        cv::subtract(x, repeatedX, x); //(x-mu)
        //generate eigenvalue matrix 5�5 populated with zeros
        cv::Mat S = cv::Mat(5,5, CV_64F, double(0));
        //set s as diagonal in it
        S = S.diag(svd.w);
        A = (x)*svd.u*S;

        ~S;
    }

    //Calculate Gaussian: y = exp(-0.5*sum(A .* A, 2)) / (sqrt(2*pi) .* prod(diag(S))) ;
    //-- A*A and perform sums by columns, to get 1 column vector
    cv::reduce(A.mul(A), A, 1,CV_REDUCE_SUM);
    cv::exp(-0.5*A,A);
    //divide A by a scalar
    double scl = (sqrt(2*CV_PI)*prodEigenVals);
    A = A/scl;
    return A;
}

cv::Mat mergePd(cv::Mat mu_d, cv::Mat c_d, cv::Mat mu_0, cv::Mat c0,cv::Mat ic0) {
    if(ic0.empty())
    {
        cv::invert(c0,ic0);
    }

    int d = c_d.rows;
    //% approximately robust
    double scl = 1e-10;
    cv::Mat icd;
    double mean = cv::mean(c_d.diag()*scl).val[0];
    cv::Mat eye;
    eye = cv::Mat::eye(d,d,CV_64F)*mean;
    cv::invert(c_d+eye,icd);
    //cv::invert((icd  + ic0)
    cv::Mat A1;
    A1 = (icd + ic0).inv();
    cv::Mat A2;
    A2 = (c_d+ eye).inv();

    cv::Mat mu;
    mu = A1*(A2*mu_d + ic0*mu_0);
    //iCd = inv(C_d+eye(d)*mean(diag(C_d))*scl) ;
    //% iC0 = inv(C_0+eye(d)*mean(diag(C_d))*scl) ;
    //% V = inv(iCd  + iC0) ;
    //% Mu = V*(iCd*Mu_d + iC0*Mu_0) ;
    //Mu = (iCd  + iC0)\((C_d+eye(d)*mean(diag(C_d))*scl)\Mu_d + iC0*Mu_0) ;

    //% nonrobust
    //% V = inv(iCd  + iC0) ;
    //% Mu = V*(inv(C_d)*Mu_d + inv(C_0)*Mu_0) ;

    return mu;
}

void loadPriorModelFromDisk(Colorspace colorSpace, std::vector<cv::Mat> &mix_Mu, std::vector<cv::Mat> &mix_Cov,
                            std::vector<cv::Mat> &mix_w, std::vector<cv::Mat> &static_prec) {
    int length = 5;
    switch (colorSpace)
    {
        case YCRCB: {
            // Mu
            double data_mu_1[] = {25, 11.1279, 178.6491, 128.4146, 124.4288};
            double data_mu_2[] = {25, 21.6993, 84.6005, 123.6754, 127.5533};
            double data_mu_3[] = {25, 38.2647, 101.2433, 126.3468, 119.7686};
            cv::Mat mix_Mu_1 = cv::Mat(length,1,CV_64F,data_mu_1).clone();
            cv::Mat mix_Mu_2 = cv::Mat(length,1,CV_64F,data_mu_2).clone();
            cv::Mat mix_Mu_3 = cv::Mat(length,1,CV_64F,data_mu_3).clone();

            mix_Mu.insert(mix_Mu.end(),mix_Mu_1);
            mix_Mu.insert(mix_Mu.end(),mix_Mu_2);
            mix_Mu.insert(mix_Mu.end(),mix_Mu_3);

            // Cov
            double  data_cov_1[] = {  209.0705, 0, 0, 0, 0,
                                      0, 46.6508, 0, 0, 0,
                                      0, 0, 986.8448, 0, 0,
                                      0, 0, 0, 30.4178, 0,
                                      0, 0, 0, 0, 26.6255};
            double data_cov_2[] = { 206.948, 0, 0, 0, 0,
                                    0, 22.4061, 0, 0, 0,
                                    0, 0, 2636.7, 0, 0,
                                    0, 0, 0, 26.2005, 0,
                                    0, 0, 0, 0, 17.7941};
            double data_cov_3[] = { 208.7815, 0, 0, 0, 0,
                                    0, 97.6139, 0, 0, 0,
                                    0, 0, 880.9302, 0, 0,
                                    0, 0, 0, 8.7445, 0,
                                    0, 0, 0, 0, 26.1301};
            cv::Mat mix_Cov_1 = cv::Mat(length,length,CV_64F,data_cov_1).clone();
            cv::Mat mix_Cov_2 = cv::Mat(length,length,CV_64F,data_cov_2).clone();
            cv::Mat mix_Cov_3 = cv::Mat(length,length,CV_64F,data_cov_3).clone();

            mix_Cov.insert(mix_Cov.end(),mix_Cov_1);
            mix_Cov.insert(mix_Cov.end(),mix_Cov_2);
            mix_Cov.insert(mix_Cov.end(),mix_Cov_3);

            // w
            mix_w.insert(mix_w.end(),3,cv::Mat(1,1,CV_64F,Scalar(0)));

            // prec
            double  data_prec_1[] = { 0.0047831, 0, 0, 0, 0,
                                      0, 0.0214358, 0, 0, 0,
                                      0, 0, 0.0010133, 0, 0,
                                      0, 0, 0, 0.0328755, 0,
                                      0, 0, 0, 0, 0.037558};
            double data_prec_2[] = {0.0048321, 0, 0, 0, 0,
                                    0, 0.0446308, 0, 0, 0,
                                    0, 0, 0.0003793, 0, 0,
                                    0, 0, 0, 0.0381672, 0,
                                    0, 0, 0, 0, 0.0561985};
            double data_prec_3[] = {0.0047897, 0, 0, 0, 0,
                                    0, 0.0102444, 0, 0, 0,
                                    0, 0, 0.0011352, 0, 0,
                                    0, 0, 0, 0.1143574, 0,
                                    0, 0, 0, 0, 0.03827};

            cv::Mat static_prec_1 = cv::Mat(length,length,CV_64F,data_prec_1).clone();
            cv::Mat static_prec_2 = cv::Mat(length,length,CV_64F,data_prec_2).clone();
            cv::Mat static_prec_3 = cv::Mat(length,length,CV_64F,data_prec_3).clone();

            static_prec.insert(static_prec.end(),static_prec_1);
            static_prec.insert(static_prec.end(),static_prec_2);
            static_prec.insert(static_prec.end(),static_prec_3);

            break;
        }
        default: // TODO: add other colorspaces
            // to lahko tudi tako da ločiš c-jevske arraye in vse ostalo. arrayi lahko ostanejo v switchu, ostalo daš ven
            break;
    }
}
void getSpacialData(cv::Size em_image_size, cv::Mat& spatial_data){
    // Spacial data mora biti: v prvi vrstici 1:50 potem pa spet 1:50 in spet 1:50  petdesetkrat...
    // V drugi vrstici pa mora biti najprej 50x 1 potem 50x 2 potem 50x 3 in tako naprej do 50.
    cv::Mat prvaVrstica_kratko(1,em_image_size.width,CV_64F);
    int i;
    for (i=0;i<em_image_size.width;i++){
        prvaVrstica_kratko.at<double>(i)=i+1.0;
    }
    // prvaVrstica_kratko je zdaj [1,2,3,...,49,50]. Zdaj moram še 50x ponovit
    cv::Mat prvaVrstica = repeat(prvaVrstica_kratko,1,em_image_size.height); // ponovitve
    // druga vrstica: najprej naredim stolpično matriko z vrednostmi [1,2,...,50]'.
    // Potem jo razširim iz enega stolpca v 50 stolpcev [1,1,1,...,1,1;2,2,2...,2,2;...;50,50,...,50]
    // Potem pa reshapeam iz 50 vrstic v 1 vrstico [1,1,...,1,1,2,2,...,2,2,....,50,5,...,50,50]
    cv::Mat drugaVrstica_stolpicni(em_image_size.height,1,CV_64F);
    for (i=0; i<em_image_size.height; i++){
        drugaVrstica_stolpicni.at<double>(i)=i+1.0;
    } // stolpicni zapolnjen
    cv::Mat drugaVrstica_matrika = repeat(drugaVrstica_stolpicni,1,em_image_size.width); // stopiram stolpec 50x
    cv::Mat drugaVrstica = drugaVrstica_matrika.reshape(0,1).clone();
    // zlepim skupaj prvo in drugo vrstico
    cv::vconcat(prvaVrstica,drugaVrstica,spatial_data);
    spatial_data = spatial_data.clone();
}
void momentMatchPdf(cv::Mat previous_Mu, cv::Mat current_Mu, cv::Mat previous_Cov, cv::Mat current_Cov, std::vector<float> current_w, cv::Mat& new_Mu, cv::Mat& new_Cov, cv::Mat& new_w){
    float sum_w=0;
    int i;
    for (i=0; i<current_w.size(); i++){
        sum_w+=current_w[i];
    }
    for (i=0; i<current_w.size(); i++){
        current_w[i]/=sum_w;
    }

    // previous_mu in current_mu sta dve stolpični matriki. Moramo ju zlepit skupaj.
    cv::Mat Mu; // za zlepljena _mu
    cv::hconcat(previous_Mu,current_Mu,Mu);

    cv::Mat Multi;
    Multi = Mat::zeros(Mu.rows, Mu.cols, Mu.type());

    for (i=0; i<Mu.cols; i++){
        Multi.col(i) = Mu.col(i)*current_w[i];
    }

    cv::Mat new_mu;
    cv::reduce(Multi, new_mu, 1, CV_REDUCE_SUM); // sešteje vsako vrstico posebej da dobi elemente novege stolpične matrike. argument 1 pomeni da bo rezultat STOLPIČNA matrika

    new_Mu = current_w[0]*current_Mu + current_w[1]*previous_Mu;

    cv::Mat temporary1, temporary2;
    temporary1 = previous_Cov+previous_Mu*previous_Mu.t();
    temporary2 = current_Cov+current_Mu*current_Mu.t();
    new_Cov = current_w[0]*temporary1 + current_w[1]*temporary2-new_mu*new_mu.t();

    new_w = cv::Mat(1,1,CV_64F,sum_w);
}
