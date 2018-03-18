#include "PipelineStage.h"
#include "Element.h"
#include "main.h"

#include <iostream>

#include <string.h>

Stage::Stage(int stageNum, std::vector<Stage> *pipeline) 
{
    //std::cerr << "<" << __FUNCTION__ << "><" << __LINE__ << ">";
    //std::cerr << stageNum << std::endl;
    
    this->stageNum_ = stageNum;
    this->queue_    = NULL;
    this->cpu_      = NULL;
    this->pipeline_ = pipeline;
}

Stage::~Stage() {}

int Stage::runStageAsync()
{
    auto rc = 0;

    //std::cerr << "<" << __FUNCTION__ << "><" << __LINE__ << ">" << std::endl;

    if (this->stageNum_ == 0) {
        // If stage is FREE
        if (this->cpu_ == NULL) {
            // Try to get ELEMENT
            this->cpu_ = new class Element(N, 1);
            //std::cerr << "<" << __FUNCTION__ << "><" << __LINE__ << ">" << std::endl;
            //this->cpu_   = this->queue_;
            /*
            if (this->queue_ == NULL) {
                std::cerr << "<" << __FUNCTION__ << "><" << __LINE__ << ">" << std::endl;
                this->queue_ = new class Element(N, 1);
                this->cpu_   = this->queue_;
                this->queue_ = NULL;
            } else {
                std::cerr << "<" << __FUNCTION__ << "><";
                std::cerr << __LINE__ << "> not null queue" << std::endl;
                exit(1);
            }
            */
        }
    // PIPELINE_STAGE_TW0 -:- PIPELINE_STAGE_FIVE
    } else {
        // If stage is FREE
        if (this->cpu_ == NULL) {
            // Try to get ELEMENT
            if (this->queue_ == NULL) {
                // If get nothing
                return 0;
            } else {
                this->cpu_ = this->queue_;
                this->queue_ = NULL;
            }
        }
    }

    // Process Element
    rc = this->cpu_->runElement(this->stageNum_);

    if (this->stageNum_ == (N-1)) {
    //if (this->stageNum_ == PIPELINE_STAGE_FIVE) {
        if (rc == ELEMENT_STAGE_COMPLETE) {
            rc = this->cpu_->getLifeTime();
            //std::cerr << "<" << rc << std::endl;
            delete (this->cpu_);
            this->cpu_ = NULL;
            // Return positive value
            return rc;
        }
    // PIPELINE_STAGE_ONE -:- PIPELINE_STAGE_FOUR
    } else {
        if (rc == ELEMENT_STAGE_COMPLETE) {
            class Element *queueNext = NULL;
            queueNext = (pipeline_->at(this->stageNum_ + 1)).queue_;
            if (queueNext == NULL) {
                pipeline_->at(this->stageNum_ + 1).queue_ = this->cpu_;
                this->cpu_ = NULL;
            } else {
                return 0;
            }
        }
    }

    return 0;
}

int Stage::runStageSync()
{
    auto rc = 0;

    //std::cerr << "<" << __FUNCTION__ << "><" << __LINE__ << ">" << std::endl;

    if (this->stageNum_ == PIPELINE_STAGE_ONE) {
        // If stage is FREE
        if (this->cpu_ == NULL) {
            // Try to get ELEMENT
            this->cpu_  = new class Element(N, 0);
        }
    // PIPELINE_STAGE_TW0 -:- PIPELINE_STAGE_FIVE
    } else {
        // If stage is FREE
        if (this->cpu_ == NULL) {
            return 0;
        }
    }

    // Process Element
    rc = this->cpu_->runElement(this->stageNum_);

    if (this->stageNum_ == (N - 1)) {
    //if (this->stageNum_ == PIPELINE_STAGE_FIVE) {
        if (rc == ELEMENT_STAGE_COMPLETE) {
            rc = this->cpu_->getLifeTime();
            delete (this->cpu_);
            this->cpu_ = NULL;
            // Return positive value
            return rc;
        }
    // PIPELINE_STAGE_ONE -:- PIPELINE_STAGE_FOUR
    } else {
        if (rc == ELEMENT_STAGE_COMPLETE) {
            class Element *queueNext = NULL;
            queueNext = (pipeline_->at(this->stageNum_ + 1)).cpu_;

            if (queueNext == NULL) {
                pipeline_->at(this->stageNum_ + 1).cpu_ = this->cpu_;
                this->cpu_ = NULL;
            } else {
                return 0;
            }
        }
    }

    return 0;
}

void Stage::showInfo()
{
    std::cout << this->stageNum_ << std::endl;
    std::cout << this->queue_ << std::endl;
    std::cout << this->cpu_ << std::endl;
    std::cout << this->pipeline_ << std::endl;
    std::cout << std::endl;
}