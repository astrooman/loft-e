#include <iostream>
#include <cstdlib>
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

void metadata::getMetaData(string metastr, int beam_num)
{
    string tmpstr;
    /*Beam number is passes as an input*/
    beam_num=beam_num;

    /***Now just manually go through get pull out values***/
    /*Target name*/
    target_name=extract_parameter_substr(metastr, "target_name:");

    /*Time stamp*/
    timestamp=extract_parameter_substr(metastr, "timestamp: ");

    /*Scan ID*/
    scan_id=stoi(extract_parameter_substr(metastr, "scan_id: "));

    /*Observing freq*/
    obs_freq=stod(extract_parameter_substr(metastr, "sky_frequency:"));

    /*Beam RA & Dec*/
    char bstr[7];
    sprintf(bstr, "beam%02d:", beam_num);
    string beamkey(bstr);
    tmpstr=extract_parameter_substr(metastr, beamkey);
    beam_ra=stod(parse_two_parameters(tmpstr, 0));
    beam_dec=stod(parse_two_parameters(tmpstr, 1));

    /*On source?*/
    on_source=stoi(extract_parameter_substr(metastr, "pk01.on_source:"));
    /*Flagged?*/
    flagged=stoi(extract_parameter_substr(metastr, "pk01.on_source:"));
}

string metadata::extract_parameter_substr(string fullmeta, string keyword)
{   
    string::size_type pos_f;
    string::size_type pos_e;
    string del_e = "\n";
    string value;
    
    /*Find position of start of keyword*/
    pos_f=fullmeta.find(keyword);
    if(pos_f != string::npos)
    {   
        pos_f+=keyword.length();
        pos_e=fullmeta.find("\n", pos_f);
        value=fullmeta.substr(pos_f, (pos_e-pos_f));
    }
    else
    {   
        cerr << "Metadata keyword: " << keyword << " not found" << endl;
    }
    /*Convert boolean parameters to 1 or 0 for later conversion to int*/
    if(value == "true") value = "1";
    if(value == "false") value = "0";
    
    return value;
}

string metadata::parse_two_parameters(string parstr, int pos)
{
    string::size_type pos_s;
    string value;

    /*Find position of start of keywork*/
    pos_s=parstr.find(" ");
    if(pos_s != string::npos)
    {
        if(pos == 0)
            value=parstr.substr(0, pos_s);
        else if(pos == 1)
            value=parstr.substr(pos_s+1, (parstr.length()-pos_s));
        else
            exit(1);
    }
    else
    {
        cerr << "Space not found " << endl;
        value = "";
    }

    return value;
}

