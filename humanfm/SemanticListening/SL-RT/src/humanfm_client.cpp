#include "humanfm_client.hpp"
#include <errno.h>

const char* HumanFMClient::SOCKET_PATH = "/tmp/humanfm.sock";

HumanFMClient::HumanFMClient() : sockfd_(-1), connected_(false) {
    // Don't connect immediately - let it connect on first use
    std::cout << "[HumanFM] Client initialized" << std::endl;
}

HumanFMClient::~HumanFMClient() {
    disconnect();
}

bool HumanFMClient::connect() {
    if (connected_) {
        return true;
    }
    
    struct sockaddr_un addr;
    
    // Create socket
    sockfd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sockfd_ == -1) {
        std::cerr << "[HumanFM] Error creating socket: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Set up address
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    
    // Connect to server
    if (::connect(sockfd_, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        std::cerr << "[HumanFM] Error connecting to socket: " << strerror(errno) << std::endl;
        close(sockfd_);
        sockfd_ = -1;
        return false;
    }
    
    connected_ = true;
    std::cout << "[HumanFM] Connected to server" << std::endl;
    return true;
}

void HumanFMClient::disconnect() {
    if (connected_ && sockfd_ != -1) {
        close(sockfd_);
        sockfd_ = -1;
        connected_ = false;
        std::cout << "[HumanFM] Disconnected from server" << std::endl;
    }
}

bool HumanFMClient::isConnected() {
    return connected_ && sockfd_ != -1;
}

std::string HumanFMClient::sendCommand(const std::string& command) {
    char buffer[HUMANFM_BUFFER_SIZE];
    
    // Try to connect if not connected
    if (!isConnected()) {
        if (!connect()) {
            std::cerr << "[HumanFM] Failed to connect for command: " << command << std::endl;
        return "";
        }
    }
    
    // Send command
    if (send(sockfd_, command.c_str(), command.length(), 0) == -1) {
        std::cerr << "[HumanFM] Error sending command: " << strerror(errno) << std::endl;
        // Connection might be broken, try to reconnect next time
        disconnect();
        return "";
    }
    
    // Receive response
    ssize_t bytesReceived = recv(sockfd_, buffer, HUMANFM_BUFFER_SIZE - 1, 0);
    if (bytesReceived == -1) {
        std::cerr << "[HumanFM] Error receiving response: " << strerror(errno) << std::endl;
        // Connection might be broken, try to reconnect next time
        disconnect();
        return "";
    }
    
    if (bytesReceived == 0) {
        std::cerr << "[HumanFM] Server closed connection" << std::endl;
        disconnect();
        return "";
    }
    
    buffer[bytesReceived] = '\0';
    return std::string(buffer);
}

std::vector<float> HumanFMClient::getClassLevels() {
    std::vector<float> levels;
    std::string response = sendCommand("getClassLevels");
    
    if (response.empty()) {
        // Return default values on error (silent fallback for robustness)
        return {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // Default to equal weights
    }
    
    try {
        Json::Value root;
        Json::Reader reader;
        
        if (!reader.parse(response, root)) {
            std::cerr << "[HumanFM] Error parsing JSON response" << std::endl;
            return {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // Default to equal weights
        }
        
        if (root.isMember("error")) {
            std::cerr << "[HumanFM] Server error: " << root["error"].asString() << std::endl;
            return {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // Default to equal weights
        }
        
        const Json::Value& data = root["data"];
        for (const auto& level : data) {
            levels.push_back(level.asFloat());
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[HumanFM] Exception parsing response: " << e.what() << std::endl;
        return {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // Default to equal weights
    }
    
    return levels;
}

std::vector<int> HumanFMClient::getClasses() {
    std::vector<int> classes;
    std::string response = sendCommand("getClasses");
    
    if (response.empty()) {
        // Return default values on error (silent fallback for robustness)
        return {1, 1, 1, 1, 1}; // Default to all classes detected
    }
    
    try {
        Json::Value root;
        Json::Reader reader;
        
        if (!reader.parse(response, root)) {
            std::cerr << "[HumanFM] Error parsing JSON response" << std::endl;
            return {1, 1, 1, 1, 1}; // Default to all classes detected
        }
        
        if (root.isMember("error")) {
            std::cerr << "[HumanFM] Server error: " << root["error"].asString() << std::endl;
            return {1, 1, 1, 1, 1}; // Default to all classes detected
        }
        
        const Json::Value& data = root["data"];
        for (const auto& cls : data) {
            classes.push_back(cls.asInt());
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[HumanFM] Exception parsing response: " << e.what() << std::endl;
        return {1, 1, 1, 1, 1}; // Default to all classes detected
    }
    
    return classes;
}

std::vector<std::string> HumanFMClient::getClassNames() {
            return {"Baby cry", "Cat", "Rooster", "Cricket", "Dog"};
} 
