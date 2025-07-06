#ifndef HUMANFM_CLIENT_HPP
#define HUMANFM_CLIENT_HPP

#include <vector>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <jsoncpp/json/json.h> // jsoncpp library

class HumanFMClient {
private:
    static const int HUMANFM_BUFFER_SIZE = 1024;
    
    // Persistent connection members
    int sockfd_;
    bool connected_;
    
    // Connection management
    bool connect();
    void disconnect();
    bool isConnected();
    
    // Helper function to send command and receive response (using persistent connection)
    std::string sendCommand(const std::string& command);
    
public:
    // Constructor and destructor for connection management
    HumanFMClient();
    ~HumanFMClient();
    // Get volume levels for the 5 target classes
    // Returns: [Speech_level, Music_level, Vehicle_level, Animal_level, Dog_level]
    std::vector<float> getClassLevels();
    
    // Get detection status for the 5 target classes  
    // Returns: [Speech_detected, Music_detected, Vehicle_detected, Animal_detected, Dog_detected]
    // Values are 0 (not detected) or 1 (detected)
    std::vector<int> getClasses();
    
    // Get class names in order
    std::vector<std::string> getClassNames();
};

#endif // HUMANFM_CLIENT_HPP 
