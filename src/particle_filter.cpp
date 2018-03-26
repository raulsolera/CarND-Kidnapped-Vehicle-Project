/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "Eigen/Dense"

#include "particle_filter.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  // Define number of particles
  num_particles = 100;

  // Helper random generator
  default_random_engine gen;
  
  // Normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // Create particles and store in vector particles
  for (int i = 0; i < num_particles; ++i) {
    
    // Create particle
    Particle _particle;
    _particle.id = i;
    _particle.x = dist_x(gen);
    _particle.y = dist_y(gen);
    _particle.theta = dist_theta(gen);
    _particle.weight = 1.0;
    
    weights.push_back(_particle.weight);
    particles.push_back(_particle);

  }

  // Initialization done!
  is_initialized = true;
  
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  
  // Helper random generator
  default_random_engine gen;
  
  // Standard normal for position x, y and theta
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  // Predict next pos of each particle
  for(int i = 0 ; i < num_particles; ++i){
    
    // predict position and yaw based on bicycle model
    double theta = particles[i].theta;
    
    if(abs(yaw_rate) < 0.0001){
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta);
    }
    
    else{
      particles[i].x += velocity/yaw_rate * (sin(theta+yaw_rate*delta_t) - sin(theta));
      particles[i].y += velocity/yaw_rate * (cos(theta) - cos(theta+yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    
    // Add noise to the prediction
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
    
  }
  
}


double euclidian_2d_distance(double x_from, double y_from, double x_to, double y_to){
  return sqrt( pow((x_from - x_to), 2.0) + pow((y_from - y_to), 2.0) );
}


void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s>landmark_list, std::vector<LandmarkObs>& observations) {
  
  // for each observation
  for(int i = 0; i < observations.size(); i++){
    
    bool first_landmark = true;
    double min_distance;
    Map::single_landmark_s nearest_landmark;
    
    // compare to each landmark and associate the nearest landmark id to observation id
    for(Map::single_landmark_s landmark : landmark_list){
      if(first_landmark){
        first_landmark = false;
        nearest_landmark = landmark;
        min_distance = euclidian_2d_distance(observations[i].x, observations[i].y,
                                             landmark.x_f, landmark.y_f);
      }
      
      else{
        double distance = euclidian_2d_distance(observations[i].x, observations[i].y,
                                                landmark.x_f, landmark.y_f);
        if(min_distance > distance){
          nearest_landmark = landmark;
          min_distance = distance;
        }
      }
    }
    observations[i].id = nearest_landmark.id_i;
  }
  
}


double multiple_gaussian(std::vector<double>measurement, std::vector<double>landmark, std::vector<double>std_dev){
  
  double factor = kOneOver2Pi;
  double exponent = 0;
  
  for(int i = 0; i < measurement.size(); i++){
    factor /= std_dev[i];
    exponent += pow(measurement[i]-landmark[i],2) / (2*pow(std_dev[i],2));
  }
  
  return factor * exp(-exponent);
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  
  // converting standard deviation in a more convenient type
  std::vector<double> std_dev;
  std_dev.push_back(std_landmark[0]);
  std_dev.push_back(std_landmark[1]);
  
  // clear particle weights vector
  weights.clear();
  
  // for each particle
  for(int i = 0; i < num_particles; i++){
    
    // new weight
    double new_weight = 1.0;
    
    // transform each observation from particle view to map coordinates
    std::vector<LandmarkObs> transformed_observations;
    
    // helpers to optimize code
    double cos_theta = cos(particles[i].theta);
    double sin_theta = sin(particles[i].theta);
    
    for(int j = 0; j < observations.size(); j++){

      // observation transformation
      LandmarkObs transformed_obs;
      transformed_obs.id = observations[j].id;
      transformed_obs.x = cos_theta * observations[j].x - sin_theta * observations[j].y + particles[i].x;
      transformed_obs.y = sin_theta * observations[j].x + cos_theta * observations[j].y + particles[i].y;
      transformed_observations.push_back(transformed_obs);
      
    }
    
    // associate each transformed observation to the closest landmark
    // call data association that will associate nearest landmark id to observation id
    // but first reduce landmarks to those within the sensor range from the particle
    std::vector<Map::single_landmark_s> landmarks_in_sensor_range;
    double distance;
    for(Map::single_landmark_s landmark : map_landmarks.landmark_list){
      distance  = euclidian_2d_distance(particles[i].x, particles[i].y, landmark.x_f, landmark.y_f);
      if(distance < 1.0 * sensor_range) landmarks_in_sensor_range.push_back(landmark);
    }
    
    dataAssociation(landmarks_in_sensor_range, transformed_observations);
    
    // calculate new weight
    for(int j = 0; j < observations.size(); j++){
      
      std::vector<double> measurement;
      measurement.push_back(transformed_observations[j].x);
      measurement.push_back(transformed_observations[j].y);
      
      std::vector<double> landmark;
      int nearest_landmark_id = transformed_observations[j].id;
      landmark.push_back(map_landmarks.landmark_list[nearest_landmark_id-1].x_f);
      landmark.push_back(map_landmarks.landmark_list[nearest_landmark_id-1].y_f);
      
      double prob = multiple_gaussian(measurement, landmark, std_dev);
      new_weight *= prob;
      
    }
    
    // update weight
    particles[i].weight = new_weight;
    weights.push_back(new_weight);
    
  }
  
}


void ParticleFilter::resample() {
  
  // resample particles
  std::vector<Particle> resample_particles;
  
  // resample particles
  random_device rd;
  mt19937 gen(rd());
  discrete_distribution<> dd(weights.begin(), weights.end());
  for(int n=0; n<num_particles; ++n) {
    resample_particles.push_back(particles[dd(gen)]);
  }
  
  particles.clear();
  particles = resample_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}