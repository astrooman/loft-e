#ifndef _H_PAFRB_GET_MJD
#define _H_PAFRB_GET_MJD

#include <iomanip>

inline double GetMjd(int ref_epoch, size_t ref_second) {

    // leap year has very stupid rules, but I don't think anyone will use
    // this code in they year 2100, when divisibility by 100 is taken into account
    // for now can simply assume, that every year that can be divided by 4 is a leap year
    short year = 2000 + (int)(ref_epoch / 2);
    short dinm[12] = {31, 28 , 31, 30 ,31, 30, 31, 31, 30, 31, 30, 31};
    if ((year % 4) == 0)
        dinm[1] = 29;
    short month = 0 + 6 * (ref_epoch % 2);
    short day = (int)(ref_second / 86400);
    int rem = ref_second - 86400 * day;
    day++;
    for (int ii = 0; ii < 6; ii++) {
        if((day - dinm[month + ii]) > 0) {
            day -= dinm[month];
            month++;
            continue;
        }
        break;
    }
    short hour = (int)(rem / 3600.0);
    short minute = (int)((rem - hour * 3600.0) / 60.0);
    short second = rem - (hour * 3600.0 + minute * 60.0);

    double jd;
    double mjd;
    month++;        // January = 1, February = 2, etc.
    int uttime = hour + ((minute + second/60.0)/60.0);
    jd = 367.0 * year - int(7.0 * (year + (int)((month + 9.0) / 12.0))/ 4.0) + (int)(275.0 * month / 9.0) + day + 1721013.5;
    mjd = jd - 2400000.5;
    double extra = ((double)hour + (((double)minute + (double)second/60.0)/60.0))/ 24.0;
    mjd += extra;

    std::ofstream mydate("mjd.txt", std::ios_base::out | std::ios_base::trunc);
    mydate.precision(6);
    mydate.setf(std::ios::fixed);
    mydate << mjd <<  " " << extra << std::endl;
    mydate.close();

    return mjd;
}

#endif
