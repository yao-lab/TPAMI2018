# Makefile

CC := g++ # This is the main compiler

# directory of objective files
BUILDDIR := build
# directory of binary files
BINDIR := bin
# directory of source code
SRCDIR := src
# extension of source file
SRCEXT := cc
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
# directory of application files
APPSDIR := apps

DEPENDENCY := $(shell find $(BUILDDIR) -type f -name *.d 2>/dev/null)

# app
Mix_feat := $(BINDIR)/parallel_lbi_with_feature

Mix_featless := $(BINDIR)/parallel_lbi_without_feature

CFLAGS := -g -std=c++0x -MMD -w


LIB := -lblas -lpthread
INC := -I include

all : $(Mix_feat) $(Mix_featless)

# APPS
##########################################################
$(Mix_feat): build/util.o build/parallel_lbi_with_feature.o
	@echo " $(CC) $^ -o $(Mix_feat) $(LIB)"; $(CC) $^ -o $(Mix_feat) $(LIB)
	@echo " $(Mix_feat) is successfully built."
	@printf '%*s' "150" | tr ' ' "-"
	@printf '\n'


$(Mix_featless): build/util.o build/parallel_lbi_without_feature.o
	@echo " $(CC) $^ -o $(Mix_featless) $(LIB)"; $(CC) $^ -o $(Mix_featless) $(LIB)
	@echo " $(Mix_featless) is successfully built."
	@printf '%*s' "150" | tr ' ' "-"
	@printf '\n'

# Compile code to objective files
###################################
$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR) $(BINDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(BUILDDIR)/%.o: $(APPSDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR) $(BINDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

# clean up the executables and objective fils
##############################################
clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(BINDIR)"; $(RM) -r $(BUILDDIR) $(BINDIR)

-include $(DEPENDENCY)

