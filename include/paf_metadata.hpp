#include <string>

using namespace std;

class metadata
{
    public:
        string timestamp;
        string target_name;
        int beam_num;
        double obs_freq;
        double beam_ra;
        double beam_dec;
        int on_source;
        int flagged;
        int scan_id;

        string extract_parameter_substr(string fullmeta, string keyword);
        string parse_two_parameters(string parstr, int pos);
        void getMetaData(string metastr, int beam_num);
};
