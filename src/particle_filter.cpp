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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Set the number of particles
	num_particles = 200;

	default_random_engine gen;
   
	// Create normal (Gaussian) distributions for x,y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]); 
  
	// Resize Particle
	particles.resize(num_particles);

	// Add random Gaussian noise to each particle and set weight to 1
	for (int i = 0; i < num_particles; ++i) {	
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);	 
		particles[i].weight = 1;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
	default_random_engine gen;

	// Add random Gaussian noise to each particle and set weight to 1
	for (int i = 0; i < num_particles; ++i) {
    
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		
		// Update x, y and the yaw angle and avoid division by zero
		if (fabs(yaw_rate) > 0.001) {
			x = x + velocity/yaw_rate * (sin(theta + yaw_rate*delta_t) - sin(theta));
			y = y + velocity/yaw_rate * (cos(theta) - cos(theta+yaw_rate*delta_t));
		}
		else {
			x = x + velocity*delta_t*cos(theta);
			y = y + velocity*delta_t*sin(theta);
		}
		theta = theta + yaw_rate*delta_t;
		
		// Create normal (Gaussian) distributions for x,y and theta
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);
		
		x = dist_x(gen);
		y = dist_y(gen);
		theta = dist_theta(gen);    
		
		// Normalize the theta to (0, 2*PI)
		while (theta > 2*M_PI) theta-=2.*M_PI;
		while (theta < 0) theta+=2.*M_PI;    
		
		particles[i].x = x;
		particles[i].y = y;
		particles[i].theta = theta;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (int i = 0; i < observations.size(); i++) {
    
		// grab current observation
		LandmarkObs observation = observations[i];

		// init minimum distance to maximum possible
		double min_dist = numeric_limits<double>::max();

		// init id of landmark from map placeholder to be associated with the observation
		int map_id = -1;
		
		for (int j = 0; j < predicted.size(); j++) {
			
			// grab current prediction
			LandmarkObs prediction = predicted[j];

			// get distance between current/predicted landmarks
			double cur_dist = dist(observation.x, observation.y, prediction.x, prediction.y);

			// find the predicted landmark nearest the current observed landmark
			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				map_id = prediction.id;
			}
		}

    // set the observation's id to the nearest predicted landmark's id
    observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	double gauss_norm, exponent, weight, x_map, y_map;

	// Reset the weight vector before updates
	weights.clear();

	for (int i = 0; i < num_particles; ++i) { 
  
		double x_part = particles[i].x;
		double y_part = particles[i].y;
		double theta = particles[i].theta;
		
		// Reset the weights for each particle before updates
		particles[i].weight = 1;
		
		for (int j = 0; j < observations.size(); ++j) {

			// Get x and y from the observations in the car coordinate 
			double x_obs = observations[j].x;
			double y_obs = observations[j].y;      

			// Transform x and y to map coordinate
			x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
			y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);

			// Find the ID of the nearest landmark based on the particle's observations
			double mu_temp = 10000;
			int mu_index = 0;
			double delta = 0;
			for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
				delta = abs(map_landmarks.landmark_list[k].x_f - x_map) + abs(map_landmarks.landmark_list[k].y_f - y_map);
				if (delta < mu_temp) {
				  mu_temp = delta;
				  mu_index = k;
				}
			}
      
			// Get the landmark position based on the ID found above
			double mu_x = map_landmarks.landmark_list[mu_index].x_f;
			double mu_y = map_landmarks.landmark_list[mu_index].y_f;

			// Calculate normalization term
			gauss_norm = (1/(2 * M_PI * sig_x * sig_y));

			// Calculate exponent
			exponent = ((x_map - mu_x) * (x_map - mu_x))/(2 * sig_x * sig_x) + ((y_map - mu_y) * (y_map - mu_y))/(2 * sig_y * sig_y);

			// Calculate weight using normalization terms and exponent
			weight = gauss_norm * exp(-exponent);

			// Multiple all the weights for each particle
			particles[i].weight *= weight;
		}
		// Update the weights vector by adding the multipled weights from each particle
		weights.push_back(particles[i].weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	
	vector<Particle> resample_particles;
	
	for (int i=0; i < num_particles; i++){
		resample_particles.push_back(particles[distribution(gen)]);
	}
	
	particles = resample_particles;	
  
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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